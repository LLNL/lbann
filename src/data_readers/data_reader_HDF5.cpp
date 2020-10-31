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
#include "lbann/data_store/data_store_conduit.hpp"
#include "conduit/conduit_schema.hpp"
#include "lbann/utils/timer.hpp"

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
  m_schema_starting_level = rhs.m_schema_starting_level;
  m_data_schema = rhs.m_data_schema;
  m_useme_schema = rhs.m_useme_schema;
  m_useme_schema_filename = rhs.m_useme_schema_filename;
}


void hdf5_data_reader::load_schema(std::string filename, conduit::Schema &schema) {
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
    conduit::Node node;
    std::string bundle_filename = ""; //XX TODO: get from sample_list
    conduit::relay::io::load(bundle_filename, "hdf5", node);
    const conduit::Schema &schema = node.schema();
    json = schema.to_json(); 
  }  
  m_comm->broadcast<std::string>(0, json, m_comm->get_world_comm());
  m_data_schema = json;
}

void hdf5_data_reader::load() {
  data_reader_sample_list::load();
  double tm1 = get_time();
  if(is_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }

  load_schema(m_useme_schema_filename, m_useme_schema);
  load_schema_from_data();

  // fills in: m_data_schema
  load_schema_from_data();

  get_datum_pathnames(m_data_schema, m_data_pathnames);
  get_datum_pathnames(m_useme_schema, m_useme_pathnames);

  validate_useme_schema();

  // Boilerplate
  m_shuffled_indices.clear();
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();

  if (is_master()) {
    std::cout << "hdf5_data_reader::load() time: " << (get_time() - tm1) << std::endl;
  }
}

void hdf5_data_reader::get_datum_pathnames(
    const conduit::Schema &schema, 
    std::unordered_set<std::string> &output,
    int n,
    std::string path) {

  int n_children = schema.number_of_children();
  if (n_children == 0) {
  std::cout << "inserting path: " << path << std::endl;
    output.insert(path);
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
  for (std::unordered_set<std::string>::const_iterator t = m_useme_pathnames.begin(); t != m_useme_pathnames.end(); t++) {
    if (m_data_pathnames.find(*t) == m_data_pathnames.end()) {
      LBANN_ERROR("you requested use of the key '", *t, ",' but that does not appear in the data's schema");
    }

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

void hdf5_data_reader::do_preload_data_store() {
  options *opts = options::get();
  double tm1 = get_time();

  // TODO: construct a more efficient owner mapping, and set it in the data store.

  hid_t file_handle;
  std::string sample_name;
  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    try {
      open_file(index, file_handle, sample_name);
      conduit::Node & node = m_data_store->get_empty_node(index);
      for (std::unordered_set<std::string>::const_iterator t = m_useme_pathnames.begin(); t != m_useme_pathnames.end(); t++) {
        if (t->find(m_metadata_field_name) == t->npos) {
      //?? TODO preload_helper(h, sample_name, m_output_scalar_prefix, index, node);
        }
      }
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

  if (get_comm()->am_world_master() ||
      (opts->get_bool("ltfb_verbose") && get_comm()->am_trainer_master())) {
    std::stringstream msg;
    msg << " loading data for role: " << get_role() << " took " << get_time() - tm1 << "s";
    LBANN_WARNING(msg.str());
  }
}

} // namespace lbann
