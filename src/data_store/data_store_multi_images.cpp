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

#include "lbann/data_store/data_store_multi_images.hpp"
#include "lbann/data_readers/data_reader_multi_images.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"



#undef DEBUG

namespace lbann {
using namespace std;

std::vector<std::string> data_store_multi_images::get_sample(size_t idx) const {
  const data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);
  data_reader_multi_images::sample_t sample = reader->get_sample(idx);
  return sample.first;
}   


void data_store_multi_images::setup() {
  double tm1 = get_time();
  if (m_rank == 0) {
    std::cerr << "starting data_store_multi_images::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }

  //sanity check
  data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);
  if (reader == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "dynamic_cast<data_reader_multi_images*>(m_reader) failed\n";
    throw lbann_exception(err.str());
  }

  m_num_img_srcs = reader->get_num_img_srcs();

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    data_store_imagenet::setup();

    if (m_rank == 0) {
      std::cerr << "data_store_multi_images setup time: " << get_time() - tm1 << std::endl;
    }
  }
}

void data_store_multi_images::get_file_sizes() {
  std::vector<Triple> my_file_sizes(m_my_datastore_indices.size()*m_num_img_srcs);
  if (m_master) std::cerr << "STARTING data_store_multi_images::get_file_sizes for " << my_file_sizes.size() << " files\n";

  size_t cur_offset = 0;
  std::unordered_map<std::string, size_t> names;
  size_t jj = 0;
  for (auto base_index : m_my_datastore_indices) {
    const std::vector<std::string> sample(get_sample(base_index));
    for (size_t k=0; k<sample.size(); k++) {
      size_t index = base_index*m_num_img_srcs + k; 
      size_t file_len = 0;
      if (names.find(sample[k]) != names.end()) {
        file_len = names[sample[k]];
      } else {
        file_len = get_file_size(m_dir, sample[k]);
        names[sample[k]] = file_len;
      }

      my_file_sizes[jj].global_index = index;
      my_file_sizes[jj].num_bytes = file_len;
      my_file_sizes[jj].offset = cur_offset;
      my_file_sizes[jj].rank = m_rank;
      cur_offset += my_file_sizes[jj].num_bytes;

      if (my_file_sizes[jj].num_bytes == 0) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
          << " file size is 0";
        throw lbann_exception(err.str());
      }
      ++jj;
    }
  }

  exchange_file_sizes(my_file_sizes, m_num_global_indices*m_num_img_srcs);
}

void data_store_multi_images::read_files() {

  #ifdef DEBUG
  // each processor opens a file and writes base_index, index, offset, and file_lenght
  // for each of its datastore indices. This was written as an aid to debugging errors
  // in one-sided communication
  char b[40];
  sprintf(b, "read_files_%d", m_rank);
  std::ofstream out(b);
  #endif 

  std::stringstream err;
  for (auto base_index : m_my_datastore_indices) {
    const std::vector<std::string> sample(get_sample(base_index));
    for (size_t k=0; k<sample.size(); k++) {
      size_t index = base_index * m_num_img_srcs + k;

      if (m_offsets.find(index) == m_offsets.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " m_offsets.find(index) failed for index: " << index;
        throw lbann_exception(err.str());
      }
      size_t offset = m_offsets[index];

      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " m_file_sizes.find(index) failed for index: " << index;
        throw lbann_exception(err.str());
      }
      size_t file_len = m_file_sizes[index];

      #ifdef DEBUG
      out << "base: " << base_index << " index: " << index << " offset: " 
          << offset << " file_len: " << file_len << "\n";;
      #endif

      if (offset + file_len > m_data.size()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
          << " of " << m_my_datastore_indices.size() << " offset: " << offset
          << " file_len: " << file_len << " offset+file_len: "
          << offset+file_len << " m_data.size(): " << m_data.size()
          << "\noffset+file_len must be <= m_data.size()";
        throw lbann_exception(err.str());
      }  

      load_file(m_dir, sample[k], m_data.data()+offset, file_len);
    }
  }
  #ifdef DEBUG
  out.close();
  #endif
}


void data_store_multi_images::extended_testing() {
  if (m_master) std::cerr << "STARTING data_store_multi_images::extended_testing()\n";
  std::stringstream err;
  std::vector<unsigned char> v;
  for (auto idx : m_my_minibatch_indices_v) {
    int base_index = (*m_shuffled_indices)[idx];
    const std::vector<std::string> sample(get_sample(base_index));
    for (size_t k=0; k<sample.size(); k++) {
      size_t index = base_index*m_num_img_srcs + k; 

      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " file length not found: " << index;
        throw lbann_exception(err.str());
      }
      size_t file_len = m_file_sizes[index];

      v.resize(file_len);
      load_file(m_dir, sample[k], v.data(), file_len);

      if (m_my_minibatch_data.find(index) == m_my_minibatch_data.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " m_my_minibatch_data.find(" << index << ") failed.";
        throw lbann_exception(err.str());
      }
      if (m_my_minibatch_data[index] != v) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " data_store_multi_images::extended_testing: "
            << " rank: " << m_rank << " index: " << index << " FAILED!\n";
        throw lbann_exception(err.str());
      }
    }
  }
  std::cerr << "rank: " << m_rank << " data_store_multi_images::extended_testing, PASSED!\n";
}

}  // namespace lbann
