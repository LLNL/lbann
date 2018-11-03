////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_store/data_store_jag.hpp"
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include <unordered_set>


namespace lbann {

typedef data_reader_jag_conduit_hdf5 conduit_reader;
//typedef data_reader_jag_conduit conduit_reader;

data_store_jag::data_store_jag(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m) {
  set_name("data_store_jag");
}

data_store_jag::~data_store_jag() {
}

void data_store_jag::setup() {
  double tm1 = get_time();
  std::stringstream err;

  if (m_master) {
    std::cerr << "starting data_store_jag::setup() for role: " 
              << m_reader->get_role() << "\n"
              << "calling generic_data_store::setup()\n";
  }
  generic_data_store::setup();

  // builds map: shuffled_index subscript -> owning proc
  build_index_owner();

  if (! m_in_memory) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    //sanity check
    conduit_reader *reader = dynamic_cast<conduit_reader*>(m_reader);
    if (reader == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<conduit_reader*>(m_reader) failed";
      throw lbann_exception(err.str());
    }
    m_jag_reader = reader;

    load_variable_names();

    if (m_master) std::cerr << "calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();
    
    if (m_master) std::cerr << "calling exchange_mb_indices()\n";
    exchange_mb_indices();

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "calling populate_datastore()\n";
    populate_datastore(); 

    if (m_master) std::cerr << "calling exchange_data()\n";
    exchange_data();
  }

  if (m_master) {
    std::cerr << "TIME for data_store_jag setup: " << get_time() - tm1 << "\n";
  }
}

void data_store_jag::get_indices(std::unordered_set<int> &indices, int p) {
  indices.clear();
  std::vector<int> &v = m_all_minibatch_indices[p];
  for (auto t : v) {
    indices.insert((*m_shuffled_indices)[t]);
  }
}


void data_store_jag::exchange_data() {
  double tm1 = get_time();
  //std::stringstream err;
  if (m_master) {
    std::cerr << "TIME for data_store_jag::exchange_data(): " 
             << get_time() - tm1 << "; role: " << m_reader->get_role() << "\n";
  }
}

void data_store_jag::populate_datastore() {
  const std::string filelist = m_reader->get_data_filename();
  std::ifstream in(filelist.c_str());
  if (!in) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + 
             " :: failed to open " + filelist + " for reading");
  }

  std::string all_filenames;

  int file_count = 0;
  std::string line;
  std::string filename;
  std::stringstream filenames;
  long int char_count = 0;
  if (m_master) {
    std::stringstream s;
    while (getline(in, line)) {
      std::stringstream s2(line);
      s2 >> filename;
      filenames << filename;
      ++file_count;
      char_count += (filename.size()+1);
    }

    all_filenames.reserve(char_count);
    while (filenames >> filename) {
      all_filenames.append(filename);
      all_filenames.append(" ");
    }
    char_count = all_filenames.size();
  }
  m_comm->world_broadcast<long int>(0, &char_count, 1);
  if (! m_master) {
    all_filenames.resize(char_count);
  }
  m_comm->world_broadcast<char>(0, &all_filenames[0], char_count);
  //m_comm->world_broadcast<const char*>(0, all_filenames.data(), char_count);

/*
  m_data.resize(m_my_datastore_indices.size());
  std::string filename;
  std::string sample_id;
  int sample_id_count = -1;
  int j = 0;
  while (getline(in, line)) {
    std::stringstream s(line);
    s >> filename;
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename );
    std::string key;
    while (s >> sample_id) {
      ++sample_id_count;
      if (m_my_datastore_indices.find(sample_id_count) != m_my_datastore_indices.end()) {
        m_data_index[sample_id_count] = j;
        std::string index = std::to_string(j);
        // load data from conduit file into m_data[j]
        conduit::Node node;
        key = sample_id + "/inputs";
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
        m_data[j][index] = node;
        for (auto scalar_name : m_scalars_to_use) {
          node.reset();
          key = sample_id + "/inputs/" + scalar_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          m_data[j][index] = node;
        }  

      ++j;
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }

  if (m_master) {
    for (size_t h=0; h<3; h++) {
      std::cout << "======================================\n";
      m_data[h].print();
    }
  }
*/
}

void data_store_jag::load_variable_names() {
#if 0
  load_inputs_to_use(m_jag_reader->m_input_keys);
  load_scalars_to_use(m_jag_reader->m_scalar_keys);
  load_image_views_to_use(m_jag_reader->m_image_views);
  load_image_channels_to_use(m_jag_reader->m_image_channels);

  if (m_master) {
    std::cerr << "using these inputs:\n";
    for (auto t : m_inputs_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\nusing these scalars:\n";
    for (auto t : m_scalars_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\nusing these views:\n";
    for (auto t : m_image_views_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\nusing these image channels: ";
    for (auto t : m_image_channels_to_use) {
      std::cerr << t << " ";
    }
    std::cerr << "\n";
  }
#endif
}

}  // namespace lbann
