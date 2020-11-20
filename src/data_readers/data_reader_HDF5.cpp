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

namespace lbann {


int hdf5_data_reader::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
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
  m_useme_schema = rhs.m_useme_schema;
  m_useme_schema_filename = rhs.m_useme_schema_filename;
}


void hdf5_data_reader::load_schema(std::string filename, conduit::Schema &schema) {
  if (filename == "") {
    LBANN_ERROR("load_schema was passed an empty filename; did you call set_schema_filename?");
  }
  std::vector<char> schema_str;
  if (is_master()) {
    load_file(filename, schema_str);
  }
  m_comm->world_broadcast(m_comm->get_world_master(), schema_str);
  std::string json_str(schema_str.begin(), schema_str.end());
  schema = json_str;
}

void hdf5_data_reader::load_schema_from_data() {
  std::string json;
  if (is_master()) {
    // Load a node, then grab the schema. This can be done better if
    // it's annoyingly slow (TODO)
    conduit::Node node;
    const sample_t& s = m_sample_list[0];
    sample_file_id_t id = s.first;
    std::stringstream ss;
    ss << m_sample_list.get_samples_dirname() << "/" 
       << m_sample_list.get_samples_filename(id);
    conduit::relay::io::load(ss.str(), "hdf5", node);
    const conduit::Schema &schema = node.schema();
    json = schema.to_json(); 
  }
  m_comm->broadcast<std::string>(0, json, m_comm->get_world_comm());

  // WARNING! CAUTION! the following is specialized for JAG;
  // may need to revisit for other data sources
  conduit::Schema schema(json);
  if (schema.number_of_children() < 1) {
      LBANN_ERROR("schema.number_of_children() < 1");
  }
  m_data_schema = schema.child(0);
}

void hdf5_data_reader::load() {
  if(is_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }
  double tm1 = get_time();

  data_reader_sample_list::load();

  // load the user's schema (i.e, specifies which data to load);
  // fills in m_useme_schema
  options *opts = options::get();
  if (!opts->has_string("schema_fn") && m_useme_schema_filename == "") {
    LBANN_ERROR("you must either include --schema_fn=<string> or call set_schema_filename");
  }
  if (opts->has_string("schema_fn")) {
    set_schema_filename(opts->get_string("schema_fn"));
  }
  load_schema(m_useme_schema_filename, m_useme_schema);

  // pull out the lbann_metadata subtree (if it exists) from the user's schema 
  if (m_useme_schema.has_child(m_metadata_field_name)) {
    m_metadata_schema = m_useme_schema[m_metadata_field_name];
    m_useme_schema.remove(m_metadata_field_name);
  }

  // load the schema from data on disk; fills in: m_data_schema 
  load_schema_from_data();

  // get ptrs to all schema nodes
  get_schema_ptrs(&m_data_schema, m_data_schema_nodes);
  get_schema_ptrs(&m_useme_schema, m_useme_schema_nodes);
  get_schema_ptrs(&m_metadata_schema, m_metadata_schema_nodes);

  // get the sets of pathnames for the data and user-supplied schemas,
  // then check that the user's schema is a subset of the data's schema
  get_datum_pathnames(m_data_schema, m_data_pathnames);
  get_datum_pathnames(m_useme_schema, m_useme_pathnames);
  validate_useme_schema();

  // fills in m_packed_to_field_names_map, calls tabulate_packing_memory_requirements
  parse_metadata();

  // may go away; for now, this reader only supports preloading mode
  opts->set_option("use_data_store", true);

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

void hdf5_data_reader::get_datum_pathnames(
    const conduit::Schema &schema, 
    std::unordered_set<std::string> &output,
    int n,
    std::string path) {
  int n_children = schema.number_of_children();
  if (n_children == 0) {
    if (output.find("#") == output.end()) {
      output.insert(path);
    }
  } 
  
  else {
    ++n;
    for (int j=0; j<n_children; j++) {
      std::stringstream extended_path;
      extended_path << path << schema.child_name(j);
      if (schema.child(j).number_of_children()) {
        extended_path << "/";
      }
      get_datum_pathnames(schema.child(j), output, n, extended_path.str());
    }
  }
}

void hdf5_data_reader::validate_useme_schema() {
  options *opts = options::get();
  bool go_deep = opts->get_bool("schema_deep_verify");
  for (std::unordered_set<std::string>::const_iterator t = m_useme_pathnames.begin(); t != m_useme_pathnames.end(); t++) {
    if (m_data_pathnames.find(*t) == m_data_pathnames.end()) {
      LBANN_ERROR("you requested use of the key '", *t, ",' but that does not appear in the data's schema");
    }

    // The following is broken (at least, I think so). 
    // Anyway, unsure if we need this level of verification;
    // if we go with using the "short" schema version, this
    // check should always fail
    go_deep = false; //TODO
    if (go_deep) {
      conduit::Schema &data_s = m_data_schema.fetch_child(*t);
      conduit::Schema &use_s = m_data_schema.fetch_child(*t);

      conduit::index_t data_s_id = data_s.dtype().id();
      conduit::index_t use_s_id = data_s.dtype().id();

      if (data_s_id != use_s_id) {
        LBANN_ERROR("data type IDs don't match");
      }
      bool success = data_s.compatible(use_s);
      if (!success) {
        LBANN_ERROR("data for this path is incompatible: ", *t);
      }
    }
  }
}

void hdf5_data_reader::do_preload_data_store() {
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "starting hdf5_data_reader::do_preload_data_store()\n";
  }

  // TODO: construct a more efficient owner mapping, and set it in the data store.

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
      LBANN_ERROR(" :: trying to load the node ", index, " and caught conduit error: ", e.what());
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

  if (is_master()) {
    std::cout << "loading data for role: " << get_role() << " took " << get_time() - tm1 << "s" << std::endl;
  }
}

// TODO: does this work? maybe not ...
int hdf5_data_reader::get_linearized_size(const std::string &key) const {
  if (m_data_pathnames.find(key) == m_data_pathnames.end()) {
    LBANN_ERROR("requested key: ", key, " is not in the schema");
  }
  conduit::DataType dt = m_data_schema.dtype();
  return dt.number_of_elements();
}

// Loads the fields that are specified in the user supplied schema.
// On entry, 'node,' which was obtained from the data_store, contains a 
// single top-level node which is the sample_id.
void hdf5_data_reader::load_sample(conduit::Node &node, size_t index) {

  // get file handle
  hid_t file_handle;
  std::string sample_name;
  data_reader_sample_list::open_file(index, file_handle, sample_name);

  // load data for the  field names that the user specified in their schema;
  // first, we load each field separately; at the end of this method
  // we call munge_data, in which we pack and or normalize, etc, the data
  for (const auto &pathname : m_useme_pathnames) {
  std::cout << "load sample; next path: " << pathname << std::endl;

    // check that the requested path exists (in the filesystem)
    const std::string original_path = "/" + sample_name + "/" + pathname;
    if (!conduit::relay::io::hdf5_has_path(file_handle, original_path)) {
      LBANN_ERROR("hdf5_has_path failed for path: ", original_path);
    }

    // get the new path-name (that contains the sample_id)
    std::stringstream ss2;
    ss2 << LBANN_DATA_ID_STR(index) << '/' << pathname;
    const std::string useme_path(ss2.str());

    // load the field's data into a conduit::Node
    conduit::relay::io::hdf5_read(file_handle, original_path, node[useme_path]);
  }

//  munge_data(node);
}

// on entry, 'node' contains data specified by the user's schema,
// with a single top-level node that contains the sample ID
void hdf5_data_reader::munge_data(conduit::Node &node) {

  // sanity check
  size_t n = node.number_of_children();
  if (n != 1) {
    LBANN_ERROR("n= ", n, "; should be 1");
  }


  // Case #1: there is no user-supplied metadata (from the user-supplied schema)
  if (m_packed_to_field_names_map.size() == 0) { 
    return;
  }

  // TODO: Case #X: normalize, etc.
  
  // Case #2: pack some or all of data
  std::unordered_set<std::string> used_field_names;
  std::vector<char> tmp_data; //temporary storage

  for (const auto t : m_packed_to_field_names_map) {
    const std::string &packed_name = t.first;
    const std::unordered_set<std::string> &field_names = t.second;
    if (m_packed_name_to_bytes.find(packed_name) == m_packed_name_to_bytes.end()) {
      LBANN_ERROR("m_packed_name_to_bytes.find(packed_name) == m_packed_name_to_bytes.end()");
    }
    size_t num_packed_bytes = m_packed_name_to_bytes[packed_name];

#if 0
    tmp_data.reserve(num_packed_bytes);
std::cout << "reserved: " << num_packed_bytes << std::endl;

    // copy data for the specified fields to tmp_data
    for (const auto &field_name : field_names) {
      if (m_field_name_to_bytes.find(field_name) == m_field_name_to_bytes.end()) {
        LBANN_ERROR("m_field_name_to_bytes.find(", field_name, ") failed");
      }
      size_t num_field_bytes = m_field_name_to_bytes[field_name];
      used_field_names.insert(field_name);
      char *v = node[packed_name].as_char8_str();
      for (size_t j=0; j<num_field_bytes; j++) {
        tmp_data.push_back(v[j]);
      }
    }

    conduit::Node &node2 = node.child(0);
    node2[packed_name] = tmp_data;
#endif
  } // for (const auto &field_name : field_names)
}

std::vector<conduit::Schema*> hdf5_data_reader::get_grand_children(conduit::Schema *schema) {
  std::vector<conduit::Schema*> grand_children;
  int n_children = schema->number_of_children();
  for (int j=0; j<n_children; j++) {
    conduit::Schema &child = schema->child(j);
    int n_grand_children = child.number_of_children();
    for (int j2=0; j2<n_grand_children; j2++) {
      grand_children.push_back(&child.child(j));
    }
  }
  return grand_children;  
}



std::vector<conduit::Schema*> hdf5_data_reader::get_children(conduit::Schema *schema) {
  std::vector<conduit::Schema*> children;
  int n_children = schema->number_of_children();
  for (int j=0; j<n_children; j++) {
    children.push_back(&schema->child(j));
  }
  return children;
}

void hdf5_data_reader::parse_metadata() {

  // if the user supplied a metadata schema, fill in m_packed_to_field_names_map
  // and m_field_name_to_packed_map
  std::vector<conduit::Schema*> metadata_operations = get_children(&m_metadata_schema);
  for (const auto &operation : metadata_operations) { //pack, cast, normalize, etc.

    if (operation->name() == "pack") {
      std::vector<conduit::Schema*> metadata_fetch = get_children(operation);
      for (const auto &fetch : metadata_fetch) { //datum, label, response
        std::vector<conduit::Schema*> metadata_field_names = get_children(fetch);
        for (const auto &field_name : metadata_field_names) { //inputs, outputs
          if (m_data_schema_nodes.find(field_name->name()) == m_data_schema_nodes.end()) {
            LBANN_ERROR("failed to find '", field_name->name(), " in m_data_schema_nodes");
          }
        
          std::vector<const conduit::Schema*> leaves;
          get_leaves(m_useme_schema_nodes[field_name->name()], leaves);
          for (const auto &leaf : leaves) {
            m_packed_to_field_names_map[operation->name()].insert(leaf->path());
            m_field_name_to_packed_map[leaf->path()] = operation->name();
          }
        }
      } // for (const auto &fetch : metadata_fetch)

      tabulate_packing_memory_requirements();
    }

    else if (operation->name() == "normalize") {
      //TODO
    }

    else if (operation->name() == "cast") {
      //TODO
    }
  } // for operation 
}

// recursive
void hdf5_data_reader::get_schema_ptrs(conduit::Schema* schema, std::unordered_map<std::string, conduit::Schema*> &schema_name_map) {
  if (schema->path() != "") {
    schema_name_map[schema->path()] = schema;
  }  
  int n_children = schema->number_of_children();
  for (int j=0; j<n_children; j++) {
    get_schema_ptrs(&schema->child(j), schema_name_map);
  }
}

// recursive
void hdf5_data_reader::get_leaves(const conduit::Schema* schema_in, std::vector<const conduit::Schema*> &leaves) {

  int n_children = schema_in->number_of_children();
  if (n_children == 0) {
    leaves.push_back(schema_in);
    return;
  } else {
    for (int j=0; j<n_children; j++) {
      get_leaves(&schema_in->child(j), leaves);
    }
  }
}

void hdf5_data_reader::tabulate_packing_memory_requirements() {

  // get the number of bytes for the data associated with each leaf;
  // note that we use leaves from the data_schema, since the useme_schema 
  // may not have data type information
  std::vector<const conduit::Schema*> data_leaves;
  get_leaves(&m_data_schema, data_leaves);
  for (const auto &leaf : data_leaves) {
    if (m_data_schema_nodes.find(leaf->path()) == m_data_schema_nodes.end()) {
      LBANN_ERROR("failed to find ", leaf->path(), " in m_data_schema_nodes");
    }
    const conduit::Schema* d_leaf = m_data_schema_nodes[leaf->path()];
    //note: leaf->path() == d_leaf->path()
    m_field_name_to_bytes[d_leaf->path()] = d_leaf->dtype().element_bytes() * d_leaf->dtype().number_of_elements();
  }
  
  // tally the total number of bytes for each operation 
  // (operations are pack, cast, normalize, etc)
  for (const auto &fetch : m_packed_to_field_names_map) {
    const std::string &packed_name = fetch.first;
    size_t total_bytes = 0;
    for (auto field_name : fetch.second) {
      if (m_field_name_to_bytes.find(field_name) == m_field_name_to_bytes.end()) {
        LBANN_ERROR("failed to find field_name=", field_name, " in m_field_name_to_bytes");
      }
      total_bytes += m_field_name_to_bytes[field_name];
    }
    m_packed_name_to_bytes[packed_name] = total_bytes;
  }
}

} // namespace lbann
