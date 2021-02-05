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

using namespace std; //XX

#undef DEBUGME
#define DEBUGME


namespace lbann {

int hdf5_data_reader::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
std::cout << " X fetch_data"; 
  return 0;
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
  m_data_schema = rhs.m_data_schema;
  m_experiment_schema = rhs.m_experiment_schema;
}

void hdf5_data_reader::load() {
  if(is_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }
  double tm1 = get_time();
  options *opts = options::get();

  // May go away; for now, this reader only supports preloading mode and
  // use of data store
  opts->set_option("preload_data_store", true);

  // Load the sample list(s)
  data_reader_sample_list::load();

  // Load and parse the schemas.
  // (yes, these are actually conduit::Nodes, but they play 
  // the part of schemas, so that's what I'm colling them)
  if (get_data_schema_filename().empty()) {
    LBANN_ERROR("you must include 'data_schema_filename' in your data reader prototext file");
  }
  if (get_experiment_schema_filename().empty()) {
    LBANN_ERROR("you must include 'data_experiment_filename' in your data reader prototext file");
  }
  load_schema(get_data_schema_filename(), m_data_schema);
  load_schema(get_experiment_schema_filename(), m_experiment_schema);
  parse_schemas();

  // the usual boilerplate (we should wrap this in a function)
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();

  if (is_master()) {
    std::cout << "hdf5_data_reader::load() time: " << (get_time() - tm1) 
              << " num samples: " << m_shuffled_indices.size() << std::endl;
  }
}

void hdf5_data_reader::load_schema(std::string filename, conduit::Node &schema) {
  // master loads the schema then bcasts to all others;
  // for now this is a world operation
  if (is_master()) {
    std::cout << "starting load_schema from file: " << filename << std::endl;
    conduit::relay::io::load(filename, schema);
  }
  conduit::relay::mpi::broadcast_using_schema(schema, m_comm->get_world_master(), m_comm->get_world_comm().GetMPIComm());
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
//      transform(node);
//      pack(node);
      m_data_store->set_preloaded_conduit_node(index, node);
    } catch (conduit::Error const& e) {
      LBANN_ERROR(" :: trying to load the node ", index, " and caught conduit exception: ", e.what());
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
             << "; global to this trainer: "<< nn <<endl;
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
cout << "starting load sample for: " << node.path()<<endl;
  // get file handle 
  hid_t file_handle;
  std::string sample_name;
  data_reader_sample_list::open_file(index, file_handle, sample_name);

  // load data for the field names specified in the user's experiment-schema
  for (const auto &p : m_useme_nodes) {
    const std::string pathname = p->path();

    // check that the requested data exists
    const std::string original_path = "/" + sample_name + "/" + pathname;
    if (!conduit::relay::io::hdf5_has_path(file_handle, original_path)) {
      LBANN_ERROR("hdf5_has_path failed for path: ", original_path);
    }

    // get the new path-name (i.e, prepend the index)
    std::stringstream ss2;
    ss2 << LBANN_DATA_ID_STR(index) << '/' << pathname;
    const std::string useme_path(ss2.str());

    cout << "XX useme path: " << useme_path << endl;
    p->print();

    if (p->child(s_metadata_node_name).has_child("coerce")) {
      cout << "XX COERCE! " << ss2.str()<<endl;
    }

    // load the field's data into the conduit::Node
    conduit::relay::io::hdf5_read(file_handle, original_path, node[useme_path]);
  }
}

// on entry, 'node' contains data specified by the user's schema,
// with a single top-level node that contains the sample ID
#if 0
void hdf5_data_reader::munge_data(conduit::Node &node) {


  // sanity check: assumption: all data has a single top-level node
  size_t n = node.number_of_children();
  if (n != 1) {
    LBANN_ERROR("n= ", n, "; should be 1");
  }

  // Return if there is no metada for normalization, etc.
  if (m_packed_to_field_names_map.size() == 0) { 
    return;
  }

  // Pack some or all of the data
  std::unordered_set<std::string> used_field_names;
  std::vector<lbann::DataType> tmp_data; //temporary storage

  for (const auto t : m_packed_to_field_names_map) {
    size_t offset = 0; // wrt tmp_data

    // allocate storage for packing data
    const std::string &packed_name = t.first;
    if (m_packed_name_to_num_elts.find(packed_name) == m_packed_name_to_num_elts.end()) {
      LBANN_ERROR("m_packed_name_to_num_elts.find(packed_name) == m_packed_name_to_num_elts.end()");
    }
    tmp_data.reserve( m_packed_name_to_num_elts[packed_name] );

    // copy data for the requested fields into tmp_data
    for (const auto &field_name : t.second) {
      if (m_field_name_to_num_elts.find(field_name) == m_field_name_to_num_elts.end()) {
        LBANN_ERROR("m_field_name_to_num_elts.find(", field_name, ") failed");
      }
      used_field_names.insert(field_name);
      try {

        // assumption: node has a single top-level child
        // that is the lbann-assigned sample id
        const conduit::Node &n3 = node.child(0);
        auto vv = n3[field_name].value();
        size_t num_elts = m_field_name_to_num_elts[field_name];
        if (static_cast<conduit::index_t>(num_elts) != n3[field_name].dtype().number_of_elements()) {
          LBANN_ERROR("this should be impossible");
        }
        const std::string &dt = n3[field_name].dtype().name();

        if (dt == "float32") {
          const float* v = n3[field_name].value();
          for (size_t k=0; k<num_elts; k++) {
            tmp_data.emplace_back(v[k]);
          }
        } else if (dt == "float64") {
          const double* v = n3[field_name].value();
          for (size_t k=0; k<num_elts; k++) {
            tmp_data.emplace_back(v[k]);
          }
        } else {
          LBANN_ERROR("Please contact Dave Hysom, or other developer, to add support for your data type: ", dt);
        }

        transform(tmp_data.data()+offset, num_elts, field_name);
        offset += num_elts;

      } catch (conduit::Error const& e) {
        LBANN_ERROR("lbann caught conduit exception: ", e.what());
      }
    } // for field_name

    // remove the data that we've packed
    conduit::Node &n4 = node.child(0);
    for (const auto &field_name : t.second) {
       n4.remove(field_name);
    }
    // TODO: think about what to do: this can leave (pun intended)
    // empty nodes and subtrees; should these be removed? Easiest
    // thing is to leave them be ...

    node[packed_name] = tmp_data;
  } // for pack_to_field_name 
}
#endif

void hdf5_data_reader::parse_schemas() {
cout<<"XXX starting parse_schemas\n";

  // these two recursive calls start at the root and "spread" the root's
  // metadata node up the branches; any non-root nodes that already contain
  // metadata will have the two nodes merged (um, sorry, thats not very clear)
  // TODO:
  adjust_metadata(&m_data_schema);
  adjust_metadata(&m_experiment_schema);

  // get pointers to all Nodes in the data schema (this is the schema for
  // the data as it resides on disk). On return, m_data_map maps:
  //       node_pathname -> Node*
  get_schema_ptrs(&m_data_schema, m_data_map);

  // Get the pathnames for the fields to be used in the current experiment;
cout << "calling get leaves multi XX\n";
  get_leaves_multi(&m_experiment_schema, m_useme_nodes);

//XX
cout << ">>>>>>>>>>>>>> useme nodes, after get_leaves_multi\n";
for (auto t : m_useme_nodes) {
  cout << "path: "<< t->path() << endl;
  t->print();
}
cout << ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<\n";
exit(0);

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

  // At this point, each Node* in m_useme_nodes:
  //   1. is a leaf node that whose values will be used in the experiment
  //   2. has a "metatdata" child node that contains instructions for
  //      munging the data, i.e: scale, bias, ordering, coesion, packing, etc.
}

// recursive. Note: this is very similar to get_leaves().
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
  // we're using this as a starting node for a search in m_data_schema.
  // (recall, m_experiment_schema is a trimmed version of m_data_schema)

  for (const auto& leaf : first) {
    std::string path_name = leaf->path();
    if (m_data_map.find(path_name) == m_data_map.end()) {
      LBANN_ERROR("pathname: ", path_name, " was not found in m_data_map");
    }

    const conduit::Node& from_meta = leaf->child(s_metadata_node_name);
    conduit::Node* node_for_recursion = m_data_map[path_name];
    // following creates the node if it doesn't exist (but it should
    // exist at this point)
    conduit::Node* to_meta = node_for_recursion->fetch_ptr(s_metadata_node_name);

    for (int i=0; i<from_meta.number_of_children(); i++) {
      const std::string& field_name = from_meta.child(i).name();
      if (!to_meta->has_child(field_name)) {
        (*to_meta)[field_name] = from_meta[field_name];
      }
    }

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

void hdf5_data_reader::tabulate_packing_memory_requirements() {
#if 0
  // load the schema from the actual data
  conduit::Schema schema;
  load_schema_from_data(schema);

  // fill in field_name -> num elts
  for (const auto &pack_name : m_packed_to_field_names_map) {
    size_t total_elts = 0;
    for (const auto &field_name : pack_name.second) {
      try {
        const conduit::Schema &s = schema.fetch_existing(field_name);
        size_t n_elts = s.dtype().number_of_elements();
        m_field_name_to_num_elts[field_name] = n_elts;
        total_elts += n_elts;
      } catch (conduit::Error const& e) {
cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.\n";
cout << "error fetching child: "<<field_name<<endl;
schema.print();
        LBANN_ERROR("caught conduit::Error: ", e.what());
      }
    }
    m_packed_name_to_num_elts[pack_name.first] = total_elts;
  }
#endif
}

void hdf5_data_reader::load_schema_from_data(conduit::Schema &schema) {
  // master loads a node and grabs the schema, then bcasts to all others
  std::string json;
  if (is_master()) {
    conduit::Node node;
    load_sample(node, 0);
    const conduit::Schema &tmp = node.schema();
    json = tmp.to_json();
  }
  m_comm->broadcast<std::string>(0, json, m_comm->get_world_comm());
  conduit::Schema data_schema(json);
  schema = data_schema.child(0);
}


void hdf5_data_reader::transform(conduit::Node& node_in) {
#if 0
  std::cout << "\nSTARTING TRANSFORM:\n";
  static bool print_warnings = true;

  // warning: possible fragility
  conduit::Node node = node_in.child(0);

  // TODO: currently I'm parsing conduit::Node* to get transform values for each
  // call; would be more efficient to cache these, if time savings warrent

  std::vector<const conduit::Node*> leaves;
  get_leaves(&node, leaves);

  const conduit::Node* metadata;
cout <<"XX iterating over "<<leaves.size()<<" leaves\n";
  for (const auto &leaf : leaves) {
    const std::string field_name = leaf->path();

    if (m_metadata_nodes.find(field_name) != m_metadata_nodes.end() && print_warnings) {
      LBANN_WARNING("failed to find metadata node for: ", field_name, "; this is not necessarily an error, but means we will not normalize or otherwise transform this datum");
    }
    metadata = m_metadata_nodes[field_name];
cout << "XX got metadata!\n";

    // Assumption: the only thing we'll ever do with scalars (aka, primitive
    // data types) is to normalize them using scale and bias; 
    // TODO relook if needed
    if (m_field_name_to_num_elts[field_name] == 1) {
      double scale = 0;
      double bias = 0;

      if (metadata->has_child("scale")) {
        scale = static_cast<double>(metadata->child("scale").value());
      }
      if (metadata->has_child("bias")) {
        bias = static_cast<double>(metadata->child("bias").value());
      }

      const std::string &dt = node[field_name].dtype().name();
      cout << "XX dt: " << dt << std::endl;

      if (dt == "float32") {
        float v = node[field_name].value();
        v = (v*scale+bias);
      } else if (dt == "float64") {
        double v = node[field_name].value();
        v = (v*scale+bias);
      } else {
      node[field_name].print();
        LBANN_ERROR("Please contact Dave Hysom, or other developer, to add support for your data type: ", dt, " with field_name: ", field_name);
      }
    }

    // Assumption: this is a 2D matrix
    else {
      //TODO: cache these values!
      try {
        const conduit::float64_array &bias = metadata->child("bias").value();
        const conduit::float64_array &scale = metadata->child("scale").value();
        const int num_channels = metadata->child("channels").value();
        const conduit::int64_array &dims = metadata->child("dims").value();
        const conduit::int64_array &xdims = metadata->child("xdims").value();
        bool hwc = false;
        if (metadata->has_child("hwc")) {
          hwc = true;
        }
        cout << "TEST: " << metadata->has_child("xdims") << " :: " << metadata->has_child("dims")<<endl;
      
      } catch (conduit::Error const& e) {
        LBANN_ERROR(" :: running transform pipeline on  matrix and caught conduit exception: ", e.what());
      }
    }
  }
  print_warnings = false;
#endif
}


/*
const std::string hdf5_data_reader::strip_sample_id(const std::string &s) {
  size_t j = s.find("/");
  if (j == std::string::npos) {
    LBANN_ERROR("failed to find '/' in string: ", s);
  }
  return s.substr(j+1);
}
*/

void hdf5_data_reader::pack(conduit::Node &node) {
cout << "xx  pack!\n";

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

} // namespace lbann
