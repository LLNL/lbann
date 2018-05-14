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


namespace lbann {

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

  set_name("data_store_multi_images");

  //sanity check
  data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);
  if (reader == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "dynamic_cast<data_reader_multi_images*>(m_reader) failed\n";
    throw lbann_exception(err.str());
  }

  m_num_img_srcs = reader->get_num_img_srcs();

  data_store_imagenet::setup();

  if (m_rank == 0) {
    std::cerr << "TIME for data_store_multi_images setup: " << get_time() - tm1 << std::endl;
  }
}

void data_store_multi_images::get_file_sizes() {
  std::vector<int> global_indices(m_my_datastore_indices.size()*m_num_img_srcs);
  std::vector<int> bytes(m_my_datastore_indices.size()*m_num_img_srcs);

  std::unordered_map<std::string, size_t> names;
  size_t jj = 0;
  size_t j = 0;
  double tm = get_time();
  for (auto base_index : m_my_datastore_indices) {
    ++j;
    if (j % 100 == 0 and m_master) {
      double e = get_time() - tm;
      double time_per_file = e / j;
      int remaining_files = (m_my_datastore_indices.size()-j)*m_num_img_srcs;
      double estimated_remaining_time = time_per_file * remaining_files;
      std::cerr << "P_0: got size for " << j*m_num_img_srcs << " of " << m_data_filepaths.size() 
                << " files; elapsed time: " << get_time() - tm 
                << "s est. remaining time: " << estimated_remaining_time << "s\n";
    }
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

      global_indices[jj] = index;
      bytes[jj] = file_len;
      ++jj;
    }
  }

  exchange_file_sizes(global_indices, bytes);
}

void data_store_multi_images::read_files(const std::unordered_set<int> &indices) {
  std::stringstream err;
  std::string local_dir = m_reader->get_local_file_dir();
  std::stringstream fp;
  double tm = get_time();
  int n = 0;
  for (auto base_index : indices) {
    ++n;
    if (n % 100 == 0 && m_master) {
      double time_per_file = (get_time() - tm) / n;
      int remaining_files = indices.size() - n;
      double estimated_remaining_time = time_per_file * remaining_files;
      std::cerr << "P_0, " << m_reader->get_role() << "; read " << n 
                << " of " << indices.size() << " files; elapsed time " 
                << (get_time() - tm)
                << "s; est. remaining time: " << estimated_remaining_time << "\n";
    }
    const std::vector<std::string> sample(get_sample(base_index));
    for (size_t k=0; k<sample.size(); k++) {
      size_t index = base_index * m_num_img_srcs + k;
      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " m_file_sizes.find(index) failed for index: " << index;
        throw lbann_exception(err.str());
      }
      if (m_data_filepaths.find(index) == m_data_filepaths.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " m_data_filepaths.find(index) failed for index: " << index;
        throw lbann_exception(err.str());
      }
      size_t file_len = m_file_sizes[index];
      fp.clear();
      fp.str("");
      fp << local_dir << "/" << m_data_filepaths[index];
      m_data[index].resize(file_len);
      load_file("", fp.str(), m_data[index].data(), file_len);
    }
  }
}

void data_store_multi_images::read_files() {
  std::stringstream err;
  for (auto base_index : m_my_datastore_indices) {
    const std::vector<std::string> sample(get_sample(base_index));
    for (size_t k=0; k<sample.size(); k++) {
      size_t index = base_index * m_num_img_srcs + k;
      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        err << __FILE__ << " " << __LINE__ << " :: " 
            << " m_file_sizes.find(index) failed for index: " << index;
        throw lbann_exception(err.str());
      }
      size_t file_len = m_file_sizes[index];
      m_data[index].resize(file_len);
      load_file(m_dir, sample[k], m_data[index].data(), file_len);
    }
  }
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


void data_store_multi_images::build_data_filepaths() {
  m_data_filepaths.clear();
  std::unordered_set<std::string> names;
  for (auto base_index : m_my_datastore_indices) {
    const std::vector<std::string> sample(get_sample(base_index));
    for (size_t k=0; k<sample.size(); k++) {
      size_t index = base_index*m_num_img_srcs + k; 
      m_data_filepaths[index] = sample[k];
    }
  }
}

} //namespace lbann

