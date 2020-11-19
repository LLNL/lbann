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


//XX for debugging during development only
void print_node_ptrs(std::unordered_map<std::string, conduit::Schema*> &s, std::string msg) {
  std::cout << "\n======================================================\n" 
            << "print_node_ptrs: " << msg << std::endl;
  for (auto t : s) {
    std::cout << "  " << t.first << std::endl;
  }
}

//XX for debugging during development only
void  print_pathnames(std::unordered_set<std::string> &p, std::string msg) {
  std::cout << "\n======================================================\n" 
            << "print_pathnames: " << msg << std::endl;
  for (auto t : p) {
    std::cout << "  " << t << std::endl;
  }
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
  get_schema_ptrs(&m_useme_schema, m_user_schema_nodes);
  get_schema_ptrs(&m_metadata_schema, m_metadata_schema_nodes);


//XX
//*
print_node_ptrs(m_data_schema_nodes, "data");
print_node_ptrs(m_user_schema_nodes, "user");
print_node_ptrs(m_metadata_schema_nodes, "metadata");
//exit(0);
//*/

  // get the sets of pathnames for the data and user-supplied schemas,
  // then check that the user's schema is a subset of the data's schema
  get_datum_pathnames(m_data_schema, m_data_pathnames);
  get_datum_pathnames(m_useme_schema, m_useme_pathnames);
  validate_useme_schema();

//XX
print_pathnames(m_data_pathnames, "data");
print_pathnames(m_useme_pathnames, "user");

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

std::cout << "\nm_data keys:\n";
for (auto &t2 : m_data_pathnames){
  std::cout << "  " << t2 << std::endl;
}
std::cout << "\nm_useme keys:\n";
for (auto &t3 : m_useme_pathnames){
  std::cout << "  " << t3 << std::endl;
}

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

//XX debug block
/*
static  bool debug = true;
if (debug) {
std::cout << index << " ==============================\n";
node.print();
debug = false;
}
*/
      m_data_store->set_preloaded_conduit_node(index, node);
    } catch (conduit::Error const& e) {
      LBANN_ERROR(" :: trying to load the node ", index, " and caught conduit error: ", e.what());
    }
  }
  /// Once all of the data has been preloaded, close all of the file handles
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

#if 0 
XX
// recursive
const conduit::Schema* find_node_by_name(const conduit::Schema* schema, const std::string &name) {
/*
std::cout << "XX  starting find_node_by_name\n";

  if (schema->name() == name) {
    return schema;
  } 

  int n_children = schema->number_of_children();
  if (n_children == 0) {
    return nullptr;
  }

  for (int j=0; j<n_children; j++) {
      get_leaves(&schema->child(j), leaves);
    }
  }
*/
}
#endif


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
std::cout << "XX STARTING load_sample\n";

  // get file handle
  hid_t file_handle;
  std::string sample_name;
  data_reader_sample_list::open_file(index, file_handle, sample_name);

  // loop over field names that the user specified in their supplied schema 
  for (const auto &pathname : m_useme_pathnames) {

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

  munge_data(node);
}

// on entry, 'node' contains data specified by the user's schema,
// with a single top-level node that contains the sample ID
void hdf5_data_reader::munge_data(conduit::Node &node) {
#if 0
std::cout << "XX STARTING hdf5_data_reader::munge_data\n";

  // sanity check
  size_t n = node.number_of_children();
  if (n != 1) {
    LBANN_ERROR("n= ", n, "; should be 1");
  }

  // Case #1: there is no user-supplied metadata (from the user-supplied schema)
  if (m_packed_to_field_names_map.size() == 0) { 
    std::cout << "XX RETURNING: no user-supplied metadata\n";
    return;
  }

  // TODO: Case #X: normalize, etc.
  
  // Case #2: pack some or all of data
  std::unordered_set<std::string> used_field_names;
  std::vector<char> tmp_data; //temporary storage

  for (const auto schema_ptr : m_packed_to_field_names_map) {
    const std::string &packed_name = t.first;
    const std::unordered_set<conduit::Schema*> &schemas = t.second;
    if (m_packed_name_to_bytes.find(packed_name) == m_packed_name_to_bytes.end()) {
      LBANN_ERROR("m_packed_name_to_bytes.find(packed_name) == m_packed_name_to_bytes.end()");
    }
    size_t num_packed_bytes = m_packed_name_to_bytes[packed_name];
if (m_packed_name_to_bytes.find(packed_name) == m_packed_name_to_bytes.end()) {
LBANN_ERROR("((");
}
std::cout << "XX num_packed_bytes: " << num_packed_bytes << std::endl;
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

  } // for (const auto &field_name : field_names)

{
std::cout << "XX -------------------------------\n";
std::cout << "XX schema after packing and before deleting paths\n";
const conduit::Schema &s = node.schema();
s.print();
}
#endif
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

std::cout << "XX STARTING parse_metadata; m_metadata_schema.name: " << m_metadata_schema.name() << "\n";

  // if the user supplied a metadata schema, fill in m_packed_to_field_names_map
  // and m_field_name_to_packed_map
  std::vector<conduit::Schema*> operations = get_children(&m_metadata_schema);
  for (const auto &operation : operations) { //pack, cast, normalize, etc.
    if (operation->name() == "pack") {
      std::vector<conduit::Schema*> op_children = get_children(operation);
      for (const auto &fetch : op_children) { //datum, label, response
        std::vector<conduit::Schema*> meta_fields = get_children(fetch);
        for (const auto &meta : meta_fields) {

          const std::string &path = meta->path();
          size_t c1 = path.find("/");
          if (c1 == std::string::npos) {
            LBANN_ERROR("failed to find '/' in string (1)");
          }
          size_t c2 = path.find("/", c1+1);
          if (c2 == std::string::npos) {
            LBANN_ERROR("failed to find '/' in string (2)");
          }
          const std::string g_child = path.substr(c2+1);

          if (m_data_schema_nodes.find(g_child) == m_data_schema_nodes.end()) {
            LBANN_ERROR("failed to find '", g_child, " in m_data_schema_nodes");
          }
        
          std::vector<const conduit::Schema*> leaves;
          get_leaves(m_data_schema_nodes[g_child], leaves);
          for (const auto &leaf : leaves) {
            m_packed_to_field_names_map[g_child].insert(leaf->path());
            m_field_name_to_packed_map[leaf->path()] = fetch->path();
          }
        }
      }

//XX
std::cout << "\nm_packed_to_field_names_map:\n";
for (auto t : m_packed_to_field_names_map) {
    std::cout << "packing name: " << t.first << std::endl;
    for (auto t3 : t.second) {
      std::cout << "  " << t3 << std::endl;
    }
}

std::cout << "\nm_field_name_to_packed_map:\n";
for (auto t : m_field_name_to_packed_map) {
  std::cout << "  " << t.first << " -> " << t.second << std::endl;
}

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

//XX
if (leaves.size() == 0) {
  std::cout << "XX  starting get_leaves; node name: " << schema_in->name() << "; path: " << schema_in->path() << "\n";
}

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
std::cout << "XX Starting tabulate_packing_memory_requirements\n";

  std::vector<const conduit::Schema*> leaves;
  get_leaves(&m_data_schema, leaves);
  for (const auto &leaf : leaves) {
    m_field_name_to_bytes[leaf->path()] = leaf->dtype().element_bytes() * leaf->dtype().number_of_elements();
  }

  std::cout << "\n==========\nm_field_name_to_bytes: \n" ;
  for (auto t : m_field_name_to_bytes) {
    std::cout << "  " << t.first << " " << t.second << std::endl;
  }
  
  for (const auto &fetch : m_packed_to_field_names_map) {
    size_t total_bytes = 0;
    for (auto field_name : fetch.second) {
      if (m_field_name_to_bytes.find(field_name) == m_field_name_to_bytes.end()) {
        LBANN_ERROR("failed to find field_name=", field_name, " in m_field_name_to_bytes");
      }

      size_t n_bytes = m_field_name_to_bytes[field_name];
      total_bytes += n_bytes;
    }
    m_packed_name_to_bytes[fetch.first] = total_bytes;
  }

  std::cout << "\n=============================\nm_packed_name_to_bytes:\n";
  for (auto t : m_packed_name_to_bytes) {
    std::cout << "  " << t.first << " " << t.second << std::endl;
  }
}

} // namespace lbann
