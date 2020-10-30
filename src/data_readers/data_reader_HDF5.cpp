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


void hdf5_data_reader::load_useme_schema() {
  if (m_useme_schema_filename == "") {
    LBANN_ERROR("you must call hdf5_data_reader::set_schema_filename(), which you apparently did not do");
  }

  std::vector<char> schema_str;
  if (is_master()) {
    load_file(m_useme_schema_filename, schema_str, false);
  }
  m_comm->world_broadcast(m_comm->get_trainer_master(), schema_str);
  std::string json_str(schema_str.begin(), schema_str.end());
  m_useme_schema = json_str;
}


void hdf5_data_reader::load() {
  data_reader_sample_list::load();
  double tm1 = get_time();
  if(is_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }

  load_useme_schema();
  get_datum_pathnames(m_useme_schema, m_useme_pathnames);

  //TODO: m_data_schema: P_0 loads a file; grabs the schema, and bcasts to others
#if 0
  get_datum_pathnames(m_data_schema, m_data_pathnames);

  validate_useme_schema();

  // Boilerplate
  m_shuffled_indices.clear();
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();
#endif
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

    std::string ss(*t);
    conduit::Schema &data_s = m_data_schema.fetch_child(ss);
    conduit::Schema &use_s = m_data_schema.fetch_child(*t);
//    const conduit::Schema &data_s = m_data_schema.fetch_existing(ss);
 //   const conduit::Schema &use_s = m_data_schema.fetch_existing(ss);

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

} // namespace lbann
