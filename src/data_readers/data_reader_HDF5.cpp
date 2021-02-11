////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
/////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_HDF5.hpp"
#include "lbann/utils/timer.hpp"
#include "conduit/conduit_relay_mpi.hpp"

#undef DEBUGME
#define DEBUGME

using namespace std;

namespace lbann {

hdf5_data_reader::hdf5_data_reader(bool shuffle) 
  : data_reader_sample_list(shuffle) {
}

hdf5_data_reader::hdf5_data_reader(const hdf5_data_reader& rhs)
  : data_reader_sample_list(rhs) {
  copy_members(rhs);
}

hdf5_data_reader& hdf5_data_reader::operator=(const hdf5_data_reader& rhs) {
  if (this == &rhs) {
    return (*this);
  }
  data_reader_sample_list::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void hdf5_data_reader::copy_members(const hdf5_data_reader &rhs) {
  m_data_schema = rhs.m_data_schema;
  m_experiment_schema = rhs.m_experiment_schema;
}

void hdf5_data_reader::load() {
  if(is_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }
  double tm1 = get_time();
  double tm11 = tm1;
  options *opts = options::get();

  if (opts->has_string("keep_packed_fields")) {
    m_delete_packed_fields = false;
  }

  // May go away; for now, this reader only supports preloading mode 
  // with data store
  opts->set_option("preload_data_store", true);

  // Load the sample list(s)
  data_reader_sample_list::load();
  if (is_master()) {
    std::cout << "time to load sample list: " << get_time() - tm11 << std::endl;
  }
  tm11 = get_time();

  // the usual boilerplate (we should wrap this in a function)
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();


  // Load and parse the user-supplied schemas
  if (get_data_schema_filename().empty()) {
    LBANN_ERROR("you must include 'data_schema_filename' in your data reader prototext file");
  }
  if (get_experiment_schema_filename().empty()) {
    LBANN_ERROR("you must include 'data_experiment_filename' in your data reader prototext file");
  }
  load_schema(get_data_schema_filename(), m_data_schema);
  load_schema(get_experiment_schema_filename(), m_experiment_schema);
  parse_schemas();
  build_useme_node_map();
  build_packing_map();
  verify_packing_data_types();

  if (is_master()) {
    std::cout << "time to load and parse the schemas: " << get_time() - tm11 
             << std::endl << "hdf5_data_reader::load() time (total): " 
              << (get_time() - tm1) 
              << " num samples: " << m_shuffled_indices.size() 
              << " for role: " << get_role() << std::endl;
  }
}

void hdf5_data_reader::build_useme_node_map() {
  for (const auto& nd : m_useme_nodes) {
    m_useme_node_map[nd->path()] = nd;
  }
}


// check that all fields that are to be packed in a group have the same
// data type
void hdf5_data_reader::verify_packing_data_types() {
  for (const auto &t : m_packing_data) {
    const std::string& group_name = t.first;
    const std::vector<PackingData>& data = t.second;
    std::string group_type;
    for (const auto& t2 : data) {
      const std::string& field_name = t2.field_name;
      std::unordered_map<std::string, conduit::Node*>::const_iterator iter = m_useme_node_map.find(field_name);
      if (iter == m_useme_node_map.end()) {
        LBANN_ERROR("failed to find path '", field_name, "' in m_useme_node_map");
      }
      const conduit::Node& metadata = (*iter->second)[s_metadata_node_name];
      if (metadata.has_child("coerce")) {
        const std::string& field_type = metadata["coerce"].as_string();
        if (group_type == "") {
          group_type = field_type;
        } else {
          if (field_type != group_type) {
            LBANN_ERROR("node ", field_name, " has type ", field_type, " but group ", group_name, " has type ", group_type);
          }
        }
      }
    }
  }
}

// master loads the use-supplied schema then bcasts to all others;
void hdf5_data_reader::load_schema(std::string filename, conduit::Node &schema) {
  if (m_comm->am_trainer_master()) {
    conduit::relay::io::load(filename, schema);
  }
  conduit::relay::mpi::broadcast_using_schema(schema, m_comm->get_trainer_master(), m_comm->get_trainer_comm().GetMPIComm());
}

// each trainer master loads a schema and bcasts to others 
void hdf5_data_reader::load_schema_from_data(conduit::Schema &schema) {
  std::string json;
  if (m_comm->am_trainer_master()) {
      size_t index = random() % m_shuffled_indices.size();
      hid_t file_handle;
      std::string sample_name;
      data_reader_sample_list::open_file(index, file_handle, sample_name);
      const std::string path = "/" + sample_name;
      if (!conduit::relay::io::hdf5_has_path(file_handle, path)) {
        LBANN_ERROR("hdf5_has_path failed for path: ", path);
      }
      conduit::Node node;
      conduit::relay::io::hdf5_read(file_handle, path, node);

      const conduit::Schema &tmp = node.schema();
      json = tmp.to_json();
      data_reader_sample_list::close_file(index);
  }

  m_comm->broadcast<std::string>(m_comm->get_trainer_master(), json, m_comm->get_trainer_comm());
  schema = json;

  if (is_master()) {
    std::cout << "schema from dataset:\n";
    schema.print();
  }
}

void hdf5_data_reader::do_preload_data_store() {
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "starting hdf5_data_reader::do_preload_data_store()\n";
  }

 for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    try {
      conduit::Node & node = m_data_store->get_empty_node(index);
      load_sample(node, index);
      m_data_store->set_preloaded_conduit_node(index, node);
    } catch (conduit::Error const& e) {
      LBANN_ERROR("trying to load the node ", index, " and caught conduit exception: ", e.what());
    }
  }
  // Once all of the data has been preloaded, close all of the file handles
  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    close_file(index);
  }

  size_t nn = m_data_store->get_num_global_indices();
  if (is_master()) {
    std::cout << "loading data for role: " << get_role() << " took " << get_time() - tm1 << "s" 
             << "num samples (local to this rank): "<< m_data_store->get_data_size()
             << "; global to this trainer: "<< nn << std::endl;
  }
}



int hdf5_data_reader::get_linearized_size(const std::string &key) const {
#if 0
  if (m_all_exp_leaveXXs.find(key) == m_data_pathnames.end()) {
    LBANN_ERROR("requested key: ", key, " is not in the schema");
  }
  conduit::DataType dt = m_data_schema.dtype();
  return dt.number_of_elements();
#endif
  return 0;
}


// Loads the fields that are specified in the user supplied schema
void hdf5_data_reader::load_sample(conduit::Node &node, size_t index) {
  hid_t file_handle;
  std::string sample_name;
  data_reader_sample_list::open_file(index, file_handle, sample_name);

  // load data for the field names specified in the user's experiment-schema
  for (const auto &p : m_useme_nodes) {
    // check that the requested data (pathname) exists on disk
    const std::string pathname = p->path();
    const std::string original_path = "/" + sample_name + "/" + pathname;
    if (!conduit::relay::io::hdf5_has_path(file_handle, original_path)) {
      LBANN_ERROR("hdf5_has_path failed for path: ", original_path);
    }

    // get the new path-name (prepend the index)
    std::stringstream ss2;
    ss2 << LBANN_DATA_ID_STR(index) << '/' << pathname;
    const std::string new_pathname(ss2.str());

    // note: this will throw an exception if the child node doesn't exist
    const conduit::Node& metadata = p->child(s_metadata_node_name);

    // load the requested data field; optionally coercing it to a different
    // data type 
    if (metadata.has_child(s_coerce_name)) {
      coerce(metadata, file_handle, original_path, new_pathname, node);
    } else {
      conduit::relay::io::hdf5_read(file_handle, original_path, node[new_pathname]);
    }

    // optionally normalize 
    if (metadata.has_child("scale")) {
      normalize(node, new_pathname, metadata);
    }
    // for images
    if (metadata.has_child("channels")) {
      repack_image(node, new_pathname, metadata);
    }
  }
  pack(node);
}


void hdf5_data_reader::normalize(
  conduit::Node& node, 
  const std::string &path, 
  const conduit::Node& metadata) {
  void* vals = node[path].data_ptr();
  size_t n_bytes = node[path].dtype().number_of_elements() * node[path].dtype().element_bytes();

  // treat this as a multi-channel image
  if (metadata.has_child("channels")) {
    // sanity check; TODO: implement for other formats when needed
    if (!metadata.has_child("hwc")) {
      LBANN_ERROR("we only currently know how to deal with HWC input images");
    }

    // get number of channels, with sanity checking
    int64_t n_channels = metadata["channels"].value();
    int sanity = metadata["scale"].dtype().number_of_elements();
    if (sanity != n_channels) {
      LBANN_ERROR("sanity: ", sanity, " should equal ", n_channels, " but instead is: ", n_channels);
    }

    // get the scale and bias arrays
    const double* scale = metadata["scale"].as_double_ptr();
    std::vector<double> b(n_channels, 0);
    const double* bias = b.data();
    if (metadata.has_child("bias")) {
      bias = metadata["bias"].as_double_ptr();
    }

    // perform the normalization
    if (node[path].dtype().is_float32()) {
      float* data = reinterpret_cast<float*>(vals);
      normalizeme<float>(data, scale, bias, n_bytes, n_channels);
    } else if  (node[path].dtype().is_float64()) {
      double* data = reinterpret_cast<double*>(vals);
      normalizeme<double>(data, scale, bias, n_bytes, n_channels);
    } else {
      LBANN_ERROR("Only float and double are currently supported for normalization");
    }
  }

  // 1D case
  else {
    double scale = metadata["scale"].value();
    double bias = 0;
    if (metadata.has_child("bias")) {
      bias = metadata["bias"].value();
    }

    if (node[path].dtype().is_float32()) {
      float* data = reinterpret_cast<float*>(vals);
      normalizeme(data, scale, bias, n_bytes);
    } else if  (node[path].dtype().is_float64()) {
      double* data = reinterpret_cast<double*>(vals);
      normalizeme(data, scale, bias, n_bytes);
    } else {
      LBANN_ERROR("Only float and double are currently supported for normalization");
    }
  }
}

void hdf5_data_reader::parse_schemas() {
  // these two recursive calls start at the root and "spread" the root's
  // metadata node up the branches; any non-root nodes that already contain
  // metadata will have the two nodes merged
  adjust_metadata(&m_data_schema);
  adjust_metadata(&m_experiment_schema);

  // get pointers to all Nodes in the data schema (this is the user-supplied 
  // schema for the data as it resides on disk). On return, m_data_map maps:
  //       node_pathname -> Node*
  get_schema_ptrs(&m_data_schema, m_data_map);

  // Get the pathnames for the fields to be used in the current experiment;
  get_leaves_multi(&m_experiment_schema, m_useme_nodes);

  // sanity checks
  std::unordered_set<std::string> sanity;
  for (const auto nd : m_useme_nodes) {
    const std::string &path =  nd->path();
    if (sanity.find(path) != sanity.end()) {
      LBANN_ERROR("sanity.find(path) != sanity.end() for path: ", path);
    }
    if (!nd->has_child(s_metadata_node_name)) {
      LBANN_ERROR("missing metadata child node for: ", path);
    }
  }

  // At this point, each object in "m_useme_nodes:"
  //   1. is a leaf node that whose values will be used in the experiment
  //   2. has a "metatdata" child node that contains instructions for
  //      munging the data, i.e: scale, bias, ordering, coersion, packing, etc.
}

// recursive
void hdf5_data_reader::get_schema_ptrs(conduit::Node* input, std::unordered_map<std::string, conduit::Node*> &schema_name_map) {

  // add the input node to the output map
  const std::string &path_name = input->path();
  if (path_name == "") {
    if (!input->is_root()) {
      LBANN_ERROR("node.path == '', but node is not root");
    }  
  } else {
    if (schema_name_map.find(path_name) != schema_name_map.end()) {
      LBANN_ERROR("duplicate pathname: ", path_name);
    }
    schema_name_map[path_name] = input;
  }

  // recurse for each child
  int n_children = input->number_of_children();
  for (int j=0; j<n_children; j++) {
    if (input->child_ptr(j)->name() != s_metadata_node_name) {
      get_schema_ptrs(&input->child(j), schema_name_map);
    }
  }
}


void hdf5_data_reader::get_leaves_multi(conduit::Node* node_in, std::vector<conduit::Node*> &leaves_out) {
  std::vector<conduit::Node*> first;
  get_leaves(node_in, first);
  // "first" contains pointers to leaf nodes from "m_experiment_schema;"
  // we use these as starting nodes for searchs in m_data_schema.
  // (recall, m_experiment_schema is a trimmed version of m_data_schema)

  for (const auto& leaf : first) {
    std::string path_name = leaf->path();
    if (m_data_map.find(path_name) == m_data_map.end()) {
      LBANN_ERROR("pathname: ", path_name, " was not found in m_data_map");
    }

    const conduit::Node& from_meta = leaf->child(s_metadata_node_name);
    conduit::Node* node_for_recursion = m_data_map[path_name];
    // "getch_ptr" creates the node if it doesn't exist (but it should
    // exist at this point)
    if (!node_for_recursion->has_child(s_metadata_node_name)) {
      LBANN_ERROR("Node with path: ", node_for_recursion->path(), " is missing its metadata node");
    }
    conduit::Node* to_meta = node_for_recursion->fetch_ptr(s_metadata_node_name);

    // update the metadata node for the seed for the next search
    // (in the data_schema) with the metadata node from the 
    // experiment_schema leaf
    for (int i=0; i<from_meta.number_of_children(); i++) {
      const std::string& field_name = from_meta.child(i).name();
      if (!to_meta->has_child(field_name)) {
        (*to_meta)[field_name] = from_meta[field_name];
      }
    }

    // recursion
    std::vector<conduit::Node*> second;
    get_leaves(m_data_map[path_name], second);
    for (auto final_leaf : second) {
      leaves_out.push_back(final_leaf);
    }
  }
}

// recursive
void hdf5_data_reader::get_leaves(conduit::Node* node, std::vector<conduit::Node*> &leaves_out) {
  // on entry, node is guaranteed to have a metadata node; but lets check:
  if (!node->has_child(s_metadata_node_name)) {
    LBANN_ERROR("missing metadata child node for node with path: ", node->path());
  }
  conduit::Node& nodes_metadata = node->child(s_metadata_node_name);

  // add fields from parent's metadata node to this node's metadata node,
  // but only if they don't already exist (existing fields takes precedence)
  if (!node->is_root()) {
    const conduit::Node &parents_metadata = node->parent()->child(s_metadata_node_name);
    if (parents_metadata.has_child(s_metadata_node_name)) {
      LBANN_ERROR("metadata nodes may not be chained");
    }
    for (int k=0; k<parents_metadata.number_of_children(); k++) {
      const std::string& field_name = parents_metadata.child(k).name();
      if (! nodes_metadata.has_child(field_name)) {
        nodes_metadata[field_name] = parents_metadata[field_name];
      }
    }
  }

  // end of recusion
  if (node->number_of_children() == 1) { // 1 is for metadata node
    leaves_out.push_back(node);
    return;
  } 

  // recursion loop
  for (int j=0; j<node->number_of_children(); j++) {
    conduit::Node* child = node->child_ptr(j);
    if (child->name() != s_metadata_node_name) {
      get_leaves(child, leaves_out);
    }
  }
}

void hdf5_data_reader::pack(conduit::Node &node) {
  for (const auto& t : m_packing_nodes) {
    const std::string& group_name = t.first;
    const std::vector<conduit::Node*> p = t.second;
    if (m_packing_types.find(group_name) == m_packing_types.end()) {
      LBANN_ERROR("failed to find group_name: ", group_name, " in m_packing_types");
    }
    const std::string& group_type = m_packing_types[group_name];
    if (group_type == "float32") {
      pack<float>(group_name, node);
    } else if (group_type == "float64") {
      pack<double>(group_name, node);
    } else {
      LBANN_ERROR("packing is currently only implemented for float32 and float64; your data type was: ", group_type, " for group_name: ", group_name);
    }
  }
}

void hdf5_data_reader::build_packing_map() {
  load_schema_from_data(m_schema_from_dataset);

  // fill in m_packing_data, which maps: group_name -> vector<PackingData>
  for (const auto &nd : m_useme_nodes) {
    const conduit::Node& metadata = (*nd)[s_metadata_node_name];
    if (metadata.has_child("pack")) {
      const std::string& group_name = metadata["pack"].as_string();
      if (!metadata.has_child("ordering")) {
        LBANN_ERROR("metadata has 'pack' but is missing 'ordering' for: ", nd->path());
      }
      conduit::int64 ordering = metadata["ordering"].value();
      const std::string& field_name = nd->path();
      int n_elts =  m_schema_from_dataset[field_name].dtype().number_of_elements();
      m_packing_data[group_name].push_back(PackingData(nd->path(), n_elts, ordering));
    }
  }

  // sort the vectors by ordering numbers
  for (auto& t : m_packing_data) {
    std::sort(t.second.begin(), t.second.end(), less_oper);
  }

  // construct the three maps that will be used during packing:
  //   m_packing_nodes, m_packing_num_elts, m_packing_types
  for (const auto& t : m_packing_data) {
    const std::string& group_name = t.first;
    const std::vector<PackingData> data = t.second;
    size_t n = 0;
    for (const auto &t2 : data) {
      n += t2.num_elts;
      std::unordered_map<std::string, conduit::Node*>::const_iterator iter = m_useme_node_map.find(t2.field_name);
      if (iter == m_useme_node_map.end()) {
        LBANN_ERROR("failed to find ", t2.field_name, " in m_useme_node_map");
      }
      m_packing_nodes[group_name].push_back(iter->second);
    }
    m_packing_num_elts[group_name] = n;
  }

  for (const auto& t : m_packing_nodes) {
    const std::string& group_name = t.first;
    const std::string& path = t.second[0]->path();
    m_packing_types[group_name] =  m_schema_from_dataset[path].dtype().name();
  }
}


// recursive
void hdf5_data_reader::adjust_metadata(conduit::Node* node) {
  //note: next call creates the node if it doesn't exist
  conduit::Node* metadata = node->fetch_ptr(s_metadata_node_name);

  if (!node->is_root()) {
    const conduit::Node* parents_metadata = node->parent()->fetch_ptr(s_metadata_node_name);
    for (int j=0; j<parents_metadata->number_of_children(); j++) {
      const std::string &field_name = parents_metadata->child(j).name();
      if (!metadata->has_child(field_name)) {
        (*metadata)[field_name] = (*parents_metadata)[field_name];
      }
    }
  }

  // recursion loop
  for (int j=0; j<node->number_of_children(); j++) {
    if (node->child_ptr(j)->name() != s_metadata_node_name) {
      adjust_metadata(node->child_ptr(j));
    }
  }
}

void hdf5_data_reader::coerce(
  const conduit::Node& metadata, 
  hid_t file_handle, 
  const std::string & original_path, 
  const std::string &new_pathname, 
  conduit::Node &node) {
  //TODO: check that we're not coercing to the same type; for now
  //      this is the user's responsibility (nothing will break)
  conduit::Node tmp;
  conduit::relay::io::hdf5_read(file_handle, original_path, tmp);

  // yay! I finally get to use a void* !!
  void* vals = tmp.data_ptr();
  size_t num_bytes = tmp.dtype().number_of_elements() * tmp.dtype().element_bytes();

  // check primitive type for data from disk
  bool from_is_float = tmp.dtype().is_float32();
  bool from_is_double = tmp.dtype().is_float64();
  if (!(from_is_float || from_is_double)) {
    LBANN_ERROR("source data is not float or data; please update the data reader");
  }

  // I don't know why, but conduit includes quotes around the string,
  // even when they're not in the json file -- so need to strip them off
  const std::string& cc = metadata[s_coerce_name].to_string();
  const std::string& coerce_to = cc.substr(1, cc.size()-2);

  // this is just ugly, but I don't know how to make it better; would
  // like a single call to coerceme (template)
  if (coerce_to == "float") {
    std::vector<float> d;
    if (from_is_float) {
      const float* from = reinterpret_cast<float*>(vals);
      coerceme(from, num_bytes, d);
    } else if (from_is_double) {
      const double* from = reinterpret_cast<double*>(vals);
      coerceme(from, num_bytes, d);
    }  
    node[new_pathname] = d;
  } else if (coerce_to == "double") {
    std::vector<double> d;
    if (from_is_float) {
      const float* from = reinterpret_cast<float*>(vals);
      coerceme(from, num_bytes, d);
    } else if (from_is_double) {
      const double* from = reinterpret_cast<double*>(vals);
      coerceme(from, num_bytes, d);
    }
    node[new_pathname] = d;
  } else {
    LBANN_ERROR("Un-implemented type requested for coercion: ", coerce_to, "; you need to update the data reader to support this");
   }
}

void hdf5_data_reader::repack_image(
  conduit::Node& node, 
  const std::string &path, 
  const conduit::Node& metadata) {

  // ==== start: sanity checking
  if (!metadata.has_child("channels")) {
    LBANN_WARNING("repack_image called, but metadata is missing the 'channels' field; please check your schemas");
    return;
  }
  if (!metadata.has_child("hwc")) {
    LBANN_ERROR("we only currently know how to deal with HWC input images");
  }
  if (!metadata.has_child("dims")) {
    LBANN_ERROR("your metadata is missing 'dims' for an image");
  }
  // ==== end: sanity checking

  void* vals = node[path].data_ptr();
  size_t n_bytes = node[path].dtype().number_of_elements() * node[path].dtype().element_bytes();
  int64_t n_channels = metadata["channels"].value();
  const conduit::int64* dims = metadata["dims"].as_int64_ptr();
  const int row_dim = dims[0];
  const int col_dim = dims[1];

  if (node[path].dtype().is_float32()) {
    float* data = reinterpret_cast<float*>(vals);
    repack_image<float>(data, n_bytes, row_dim, col_dim, n_channels);
  } else if (node[path].dtype().is_float64()) {
    double* data = reinterpret_cast<double*>(vals);
    repack_image<double>(data, n_bytes, row_dim, col_dim, n_channels);
  } else {
    LBANN_ERROR("Only float and double are currently supported for normalization");
  }
}


//==========================================================================
// the following methods are included for testing and backwards compatibility;
// they may (should?) go away in the future
//
int hdf5_data_reader::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
std::cerr << " X fetch_data"; 
  return 0;
}

} // namespace lbann
