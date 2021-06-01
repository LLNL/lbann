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

namespace lbann {

hdf5_data_reader::~hdf5_data_reader() {
}

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
  m_data_dims_lookup_table = rhs.m_data_dims_lookup_table;
  m_linearized_size_lookup_table = rhs.m_linearized_size_lookup_table;
  m_experiment_schema_filename = rhs.m_experiment_schema_filename;
  m_data_schema_filename = rhs.m_data_schema_filename;
  m_delete_packed_fields = rhs.m_delete_packed_fields;
  m_packing_groups = rhs.m_packing_groups;
  m_experiment_schema = rhs.m_experiment_schema;
  m_data_schema = rhs.m_data_schema;
  m_useme_node_map = rhs.m_useme_node_map;
  //m_data_map should not be copied, as it contains pointers, and is only
  //needed for setting up other structures during load

  if(rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
    m_data_store->set_data_reader_ptr(this);
  }
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

  if (is_master()) {
    std::cout << "time to load and parse the schemas: " << get_time() - tm11
              << std::endl << "hdf5_data_reader::load() time (total): "
              << (get_time() - tm1)
              << "\nnum samples: " << m_shuffled_indices.size()
              << "\nfor role: " << get_role() << std::endl;
  }

  if (!opts->get_bool("quiet") && is_master()) {
    print_metadata();
  }
}

// master loads the experiment-schema then bcasts to others
void hdf5_data_reader::load_schema(std::string filename, conduit::Node &schema) {
  if (m_comm->am_trainer_master()) {
    conduit::relay::io::load(filename, schema);
  }
  conduit::relay::mpi::broadcast_using_schema(schema, m_comm->get_trainer_master(), m_comm->get_trainer_comm().GetMPIComm());
}

void hdf5_data_reader::do_preload_data_store() {
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "starting hdf5_data_reader::do_preload_data_store() for role: " << get_role() << std::endl;
  }

  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != get_rank()) {
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
    if(m_data_store->get_index_owner(index) != get_rank()) {
      continue;
    }
    close_file(index);  //data_reader_sample_list::close_file
  }

  size_t nn = m_data_store->get_num_global_indices();
  if (is_master()) {
    std::cout << "loading data for role: " << get_role() << " took " << get_time() - tm1 << "s"
             << "num samples (local to this rank): "<< m_data_store->get_data_size()
             << "; global to this trainer: "<< nn << std::endl;
  }

}

// Loads the fields that are specified in the user supplied schema
void hdf5_data_reader::load_sample(conduit::Node &node, size_t index, bool ignore_failure) {
  hid_t file_handle;
  std::string sample_name;

  data_reader_sample_list::open_file(index, file_handle, sample_name);
  // load data for the field names specified in the user's experiment-schema
  for (const auto &p : m_useme_node_map) {
    // do not load a "packed" field, as it doesn't exist on disk!
    if (! is_composite_node(p.second)) {

      // check that the requested data (pathname) exists on disk
      const std::string pathname = p.first;
      const std::string original_path = "/" + sample_name + "/" + pathname;
      if (!conduit::relay::io::hdf5_has_path(file_handle, original_path)) {
        if (ignore_failure) {
          continue;
        }
        LBANN_ERROR("hdf5_has_path failed for path: ", original_path);
      }

      // get the new path-name (prepend the index)
      std::stringstream ss2;
      ss2 << LBANN_DATA_ID_STR(index) << '/' << pathname;
      const std::string new_pathname(ss2.str());

      // note: this will throw an exception if the child node doesn't exist
      const conduit::Node& metadata = p.second.child(s_metadata_node_name);

      // optionally coerce the data, e.g, from double to float, per settings
      // in the experiment_schema
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
      if (metadata.has_child("channels") && metadata["channels"].as_int64() > 1) {
        repack_image(node, new_pathname, metadata);
      }
    }
  }

  pack(node, index);
}

void hdf5_data_reader::normalize(
  conduit::Node& node,
  const std::string &path,
  const conduit::Node& metadata) {
  void* vals = node[path].data_ptr();
  size_t n_bytes = node[path].dtype().number_of_elements() * node[path].dtype().element_bytes();

  // treat this as a multi-channel image
  if (metadata.has_child("channels")) {

    // get number of channels, with sanity checking
    int64_t n_channels = metadata["channels"].value();
    int sanity = metadata["scale"].dtype().number_of_elements();
    if (sanity != n_channels) {
      LBANN_ERROR("sanity: ", sanity, " should equal ", n_channels, " but instead is: ", n_channels);
    }

    // sanity check; TODO: implement for other formats when needed
    if (n_channels > 1 && !metadata.has_child("hwc")) {
      LBANN_ERROR("we only currently know how to deal with HWC input images");
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
  adjust_metadata(&m_data_schema);
  adjust_metadata(&m_experiment_schema);

  // get pointers to all Nodes in the data schema (this is the user-supplied
  // schema for the data as it resides on disk). On return, m_data_map maps:
  //       node_pathname -> Node*
  get_schema_ptrs(&m_data_schema, m_data_map);

  // Get the pathnames for the fields to be used in the current experiment;
  get_leaves_multi(&m_experiment_schema, m_useme_node_map_ptrs);

  // sanity checks
  std::unordered_set<std::string> sanity;
  for (const auto nd : m_useme_node_map_ptrs) {
    const std::string &path =  nd.first;
    if (sanity.find(path) != sanity.end()) {
      LBANN_ERROR("sanity.find(path) != sanity.end() for path: ", path);
    }
    if (!nd.second->has_child(s_metadata_node_name)) {
      LBANN_ERROR("missing metadata child node for: ", path);
    }
  }

  // At this point, each object in "m_useme_node_map_ptrs:"
  //   1. is a leaf node whose values will be used in the experiment
  //   2. has a "metatdata" child node that contains instructions for
  //      munging the data, i.e: scale, bias, ordering, coersion, packing, etc.

  // ptrs were a problem when carving out a validation set ...
  for (const auto &t : m_useme_node_map_ptrs) {
    m_useme_node_map[t.first] = *t.second;
  }
  construct_linearized_size_lookup_tables();
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


void hdf5_data_reader::get_leaves_multi(
    conduit::Node* node_in,
    std::unordered_map<std::string, conduit::Node*> &leaves_out,
    bool ignore_metadata) {
  std::unordered_map<std::string, conduit::Node*> first;
  get_leaves(node_in, first, ignore_metadata);

  // "first" contains pointers to leaf nodes from "m_experiment_schema;"
  // we use these as starting nodes for searchs in m_data_schema.
  // (recall, m_experiment_schema is a trimmed version of m_data_schema)

  for (const auto& leaf : first) {
    std::string path_name = leaf.first;
    if (m_data_map.find(path_name) == m_data_map.end()) {
      LBANN_ERROR("pathname: ", path_name, " was not found in m_data_map; num map entries: ", m_data_map.size(), "; role: ", get_role());
    }

    const conduit::Node& from_meta = leaf.second->child(s_metadata_node_name);
    conduit::Node* node_for_recursion = m_data_map[path_name];
    // "fetch_ptr" creates the node if it doesn't exist (though it should
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
    std::unordered_map<std::string, conduit::Node*> second;
    get_leaves(m_data_map[path_name], second);
    for (auto t : second) {
      leaves_out[t.first] = t.second;
    }
  }
}

// recursive
void hdf5_data_reader::get_leaves(conduit::Node* node, std::unordered_map<std::string, conduit::Node*> &leaves_out, bool ignore_metadata) {

  // merge fields from parent's metadata node to this node's metadata node,
  // but only if they don't already exist (existing fields takes precedence)
  if (!ignore_metadata) {
    if (!node->has_child(s_metadata_node_name)) {
    LBANN_ERROR("missing metadata child node for node with path: ", node->path());
    }
    conduit::Node& metadata = node->child(s_metadata_node_name);
    if (!node->is_root()) {
      const conduit::Node &parents_metadata = node->parent()->child(s_metadata_node_name);
      if (parents_metadata.has_child(s_metadata_node_name)) {
       LBANN_ERROR("metadata nodes may not be chained");
      }
      for (int k=0; k<parents_metadata.number_of_children(); k++) {
        const std::string& field_name = parents_metadata.child(k).name();
        if (! metadata.has_child(field_name)) {
          metadata[field_name] = parents_metadata[field_name];
        }
      }
    }
  } // end, deal with metadata

  // end of recusion conditions: no children, or only child is "metadata"
  int n = node->number_of_children();
  if (n == 0) {
    leaves_out[node->path()] =  node ;
    return;
  }
  if (n == 1 && node->child(0).name() == s_metadata_node_name) {
    leaves_out[node->path()] = node ;
    return;
  }

  // recursion loop
  for (int j=0; j<node->number_of_children(); j++) {
    conduit::Node* child = node->child_ptr(j);
    if (child->name() != s_metadata_node_name) {
      get_leaves(child, leaves_out, ignore_metadata);
    }
  }
}

void hdf5_data_reader::pack(conduit::Node &node, size_t index) {
  if (m_packing_groups.size() == 0) {
    build_packing_map(node.child(0));
  }
  for (const auto& t : m_packing_groups) {
    const std::string& group_name = t.first;
    const PackingGroup& g = t.second;
    std::string group_type = conduit::DataType::id_to_name(g.data_type);
    if (group_type == "float32") {
      pack<float>(group_name, node, index);
    } else if (group_type == "float64") {
      pack<double>(group_name, node, index);
    } else {
      LBANN_ERROR("packing is currently only implemented for float32 and float64; your data type was: ", group_type, " for group_name: ", group_name);
    }
  }
}

struct PackingData {
    PackingData(std::string s, int n_elts, size_t dt, int order)
      : field_name(s), num_elts(n_elts), dtype(dt), ordering(order) {}
    PackingData() {}
    std::string field_name;
    int num_elts;
    size_t dtype;
    conduit::index_t ordering;
};

struct {
  bool operator()(const PackingData& a, const PackingData& b) const {
    return a.ordering < b.ordering;
  }
} less_oper;

void hdf5_data_reader::build_packing_map(conduit::Node &node) {
  std::unordered_map<std::string, std::vector<PackingData>> packing_data;
  for (const auto& nd : m_useme_node_map_ptrs) {
    const conduit::Node& metadata = (*nd.second)[s_metadata_node_name];
    if (metadata.has_child("pack")) {
      const std::string& group_name = metadata["pack"].as_string();
      if (!metadata.has_child("ordering")) {
        LBANN_ERROR("metadata has 'pack' but is missing 'ordering' for: ", nd.first);
      }
      conduit::int64 ordering = metadata["ordering"].value();
      const std::string& field_name = nd.first;
      int n_elts =  node[field_name].dtype().number_of_elements();
      size_t data_type =  node[field_name].dtype().id();
      packing_data[group_name].push_back(PackingData(field_name, n_elts, data_type, ordering));
    }
  }

  // sort the vectors by ordering numbers
  for (auto& t : packing_data) {
    std::sort(t.second.begin(), t.second.end(), less_oper);
  }

  for (const auto& t : packing_data) {
    const std::string& group_name = t.first;
    m_packing_groups[group_name].group_name = group_name;  // ACH!
    for (const auto& t2 : t.second) {
      m_packing_groups[group_name].names.push_back(t2.field_name);
      m_packing_groups[group_name].sizes.push_back(t2.num_elts);
      m_packing_groups[group_name].data_types.push_back(t2.dtype);
    }
    size_t n_elts = 0;
    conduit::index_t id_sanity = 0;
    for (size_t k=0; k<m_packing_groups[group_name].names.size(); k++) {
      n_elts += m_packing_groups[group_name].sizes[k];
      if (id_sanity == 0) {
        id_sanity = m_packing_groups[group_name].data_types[k];
      } else {
        if (m_packing_groups[group_name].data_types[k] != id_sanity) {
          LBANN_ERROR("m_packing_groups[group_name].data_types[k] != id_sanity; you may need to coerce a data type in your schema");
        }
      }
    }
    m_packing_groups[group_name].n_elts = n_elts;
    m_packing_groups[group_name].data_type = id_sanity;
  }
}


// Union your parent's metadata with yours; in case of
// a key conflict, the child's values prevail
// (recursive)
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
  conduit::Node tmp;
  conduit::relay::io::hdf5_read(file_handle, original_path, tmp);

  // yay! I finally get to use a void*
  void* vals = tmp.data_ptr();
  size_t num_bytes = tmp.dtype().number_of_elements() * tmp.dtype().element_bytes();

  // get data type for data from disk
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
  // like to have a single call to coerceme<>
  if (coerce_to == "float") {
    std::vector<float> d;
    if (from_is_float) {
      const float* from = reinterpret_cast<float*>(vals);
      coerceme<float>(from, num_bytes, d);
    } else if (from_is_double) {
      const double* from = reinterpret_cast<double*>(vals);
      coerceme<double>(from, num_bytes, d);
    }
    node[new_pathname] = d;
  } else if (coerce_to == "double") {
    std::vector<double> d;
    if (from_is_float) {
      const float* from = reinterpret_cast<float*>(vals);
      coerceme<float>(from, num_bytes, d);
    } else if (from_is_double) {
      const double* from = reinterpret_cast<double*>(vals);
      coerceme<double>(from, num_bytes, d);
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

const std::vector<int> hdf5_data_reader::get_data_dims(std::string name) const {
  std::unordered_map<std::string, std::vector<int>>::const_iterator iter = m_data_dims_lookup_table.find(name);
  if (iter == m_data_dims_lookup_table.end()) {
    LBANN_ERROR("get_data_dims_size was asked for info about an unknown field name: ", name);
  }
  return iter->second;
}

int hdf5_data_reader::get_linearized_size(std::string name) const {
  std::unordered_map<std::string, int>::const_iterator iter = m_linearized_size_lookup_table.find(name);
  if (iter == m_linearized_size_lookup_table.end()) {
    LBANN_ERROR("get_linearized_data_size was asked for info about an unknown field name: ", name, "; table size: ", m_linearized_size_lookup_table.size(), " for role: ", get_role());
  }
  return iter->second;
}

//fills in: m_data_dims_lookup_table and m_linearized_size_lookup_table;
void hdf5_data_reader::construct_linearized_size_lookup_tables() {
  m_linearized_size_lookup_table.clear();
  m_data_dims_lookup_table.clear();

  conduit::Node node;
  size_t index = random() % m_shuffled_indices.size();
  load_sample(node, index);
  std::unordered_map<std::string, conduit::Node*> leaves;
  get_leaves(&node, leaves, true);

  std::stringstream ss;
  ss << '/' << LBANN_DATA_ID_STR(0);
  size_t sz = ss.str().size();

  // loop over the data-fields (aka, leaves)
  for (const auto& t : leaves) {
      const std::string field_name = t.first.substr(sz);
    // we only care about leaves that are being used in the current experiment
    if (m_useme_node_map_ptrs.find(field_name) != m_useme_node_map_ptrs.end()) {

      // add entry to linearized size lookup table
      size_t n_elts = t.second->dtype().number_of_elements();
      m_linearized_size_lookup_table[field_name] = n_elts;

      // remainder of this block is filling in the m_data_dims_lookup_table
      conduit::Node* nd = m_useme_node_map_ptrs[field_name];
      conduit::Node* metadata = nd->fetch_ptr(s_metadata_node_name);

      // easy case: scalars, vectors
      if (!metadata->has_child("channels")) {
        m_data_dims_lookup_table[field_name].push_back(n_elts);
      }

      // error prone case; depends on user correctly writing schema
      // data dims for JAG images are: {4, 64, 64}; they may have previously
      // been {64, 64}; this could be a problem
      else {
        int channels = metadata->child("channels").to_int32();
        m_data_dims_lookup_table[field_name].push_back(channels);
        int nn_elts = metadata->child("dims").dtype().number_of_elements();
        const conduit::int64* tmp = metadata->child("dims").as_int64_ptr();
        for (int k=0;k<nn_elts;k++) {
          m_data_dims_lookup_table[field_name].push_back(tmp[k]);
        }
      }
    }
  }
}

bool hdf5_data_reader::fetch(std::string which, CPUMat& Y, int data_id, int mb_idx) {
  size_t n_elts = 0;
  std::string dtype;
  const void* d = get_data(data_id, which, n_elts, dtype);

  if (dtype == "float64") {
    const conduit::float64* data = reinterpret_cast<const conduit::float64*>(d);
    for (size_t j = 0; j < n_elts; ++j) {
      Y(j, mb_idx) = data[j];
    }
  } else if (dtype == "float32") {
    const conduit::float32* data = reinterpret_cast<const conduit::float32*>(d);
    for (size_t j = 0; j < n_elts; ++j) {
      Y(j, mb_idx) = data[j];
    }
  } else if (dtype == "int64") {
    const conduit::int64* data = reinterpret_cast<const conduit::int64*>(d);
    for (size_t j = 0; j < n_elts; ++j) {
      Y(j, mb_idx) = data[j];
    }
  } else if (dtype == "int32") {
    const conduit::int32* data = reinterpret_cast<const conduit::int32*>(d);
    for (size_t j = 0; j < n_elts; ++j) {
      Y(j, mb_idx) = data[j];
    }
  } else if (dtype == "uint64") {
    const conduit::uint64* data = reinterpret_cast<const conduit::uint64*>(d);
    for (size_t j = 0; j < n_elts; ++j) {
      Y(j, mb_idx) = data[j];
    }
  } else if (dtype == "uint32") {
    const conduit::uint32* data = reinterpret_cast<const conduit::uint32*>(d);
    for (size_t j = 0; j < n_elts; ++j) {
      Y(j, mb_idx) = data[j];
    }
  } else {
    LBANN_ERROR("unknown dtype: ", dtype);
  }

  return true;
}

void hdf5_data_reader::print_metadata(std::ostream& os) {
  os << std::endl << "Metadata and data-type information for all loaded fields follows, for role: " << get_role() << std::endl;

  // load a sample from file, applying all transformations along the way;
  // need to do this so we can get the correct dtypes
  conduit::Node populated_node;
  size_t index = random() % m_shuffled_indices.size();
  bool ignore_failure = true;
  load_sample(populated_node, index, ignore_failure);

  // get all leaves (data fields)
  std::unordered_map<std::string, conduit::Node*> leaves;
  get_leaves(&populated_node, leaves, true);

  // build map: field_name -> Node
  std::unordered_map<std::string,conduit::Node*> mp;
  for (const auto& t : leaves) {
    size_t j = t.first.find('/');
    mp[t.first.substr(j+1)] = t.second;
  }

  // print metadata and data types for all other nodes
  for (const auto& t : m_useme_node_map_ptrs) {
    const std::string& name = t.first;
    conduit::Node* node = t.second;
    os << "==================================================================\n"
       << "field: " << t.first << std::endl;
    if (!node->has_child(s_metadata_node_name)) {
      LBANN_ERROR("missing metadata node for ", name);
    }
    conduit::Node& metadata = node->fetch_existing(s_metadata_node_name);
    std::string yaml_metadata = metadata.to_yaml();
    os << "\nMetadata Info:\n--------------" << yaml_metadata;

    if (mp.find(name) != mp.end()) {
      const conduit::Node* nd = mp[name];
      std::string yaml_dtype = nd->dtype().to_yaml();
    os << "\nDataType Info:\n--------------\n" << yaml_dtype;
    } else  {
      LBANN_ERROR("mp.find(", name, " failed");
    }
    os << std::endl;
  }
  os << "==================================================================\n\n";
}

bool hdf5_data_reader::is_composite_node(const conduit::Node& node) {
  if (!node.has_child(s_metadata_node_name)) {
    LBANN_ERROR("node with path: ", node.path(), " is missing metadata child node");
  }
  const conduit::Node& metadata = node.child(s_metadata_node_name);
  return metadata.has_child(s_composite_node);
}

void hdf5_data_reader::set_data_schema(const conduit::Schema& s) {
  m_data_schema = s;
  parse_schemas();
}

void hdf5_data_reader::set_experiment_schema(const conduit::Schema& s) {
  m_experiment_schema = s;
  parse_schemas();
}

//Note to developers and reviewer: this is very conduit-ishy; I keep thinking
//there's a simpler, more elegant way to do this, but I'm not seeing it.
const void* hdf5_data_reader::get_data(
    const size_t sample_id_in,
    std::string field_name_in,
    size_t &num_elts_out,
    std::string& dtype_out) const {

  // get the pathname to the data, and verify it exists in the conduit::Node
  const conduit::Node& node = m_data_store->get_conduit_node(sample_id_in);
  std::stringstream ss;
  ss << node.name() << node.child(0).name() + "/" << field_name_in;
  if (!node.has_path(ss.str())) {
    LBANN_ERROR("no path: ", ss.str());
  }

  num_elts_out = node[ss.str()].dtype().number_of_elements();

  const void* r;
  dtype_out = node[ss.str()].dtype().name();
  if (dtype_out == "float64") {
    r = reinterpret_cast<const void*>(node[ss.str()].as_float64_ptr());
  } else if (dtype_out == "float32") {
    r = reinterpret_cast<const void*>(node[ss.str()].as_float32_ptr());
  } else if (dtype_out == "int64") {
    r = reinterpret_cast<const void*>(node[ss.str()].as_int64_ptr());
  } else if (dtype_out == "int32") {
    r = reinterpret_cast<const void*>(node[ss.str()].as_int32_ptr());
  } else if (dtype_out == "uint64") {
    r = reinterpret_cast<const void*>(node[ss.str()].as_uint64_ptr());
  } else if (dtype_out == "uint32") {
    r = reinterpret_cast<const void*>(node[ss.str()].as_uint32_ptr());
  } else {
    LBANN_ERROR("unknown dtype; not float32/64, int32/64, or uint32/64; dtype is reported to be: ", dtype_out);
  }
  return r;
}

} // namespace lbann
