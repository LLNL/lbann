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
#include "lbann/data_readers/data_reader_triplet.hpp"
#include "lbann/data_readers/data_reader_multi_images.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include <sys/stat.h>

namespace lbann {

std::unordered_map<std::string, size_t> name_2_idx;
std::unordered_map<size_t, std::string> idx_2_name;
std::unordered_map<std::string, size_t> name_2_size;

void data_store_multi_images::setup(bool test_dynamic_cast, bool run_tests) {
  double tm1 = get_time();
  if (m_rank == 0) {
    std::cerr << "starting data_store_multi_images::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }

  //sanity check
  data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);
  if (reader == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "dynamic_cast<data_reader_multi_images*>(m_reader) failed";
    throw lbann_exception(err.str());
  }

  m_num_img_srcs = reader->get_num_img_srcs();

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
    //m_buffers.resize( omp_get_max_threads() );
  } 
  
  else {
    data_store_imagenet::setup(false, false);

    if (options::get()->has_bool("test_data_store") && options::get()->get_bool("test_data_store") && run_tests) {
      options::get()->set_option("exit_after_setup", true);
      test_data();
    }


    double tm2 = get_time();
    if (m_rank == 0) {
      std::cerr << "data_store_multi_images setup time: " << tm2 - tm1 << std::endl;
    }
  }
}

void data_store_multi_images::get_file_sizes() {
  std::vector<Triple> my_file_sizes(m_my_global_indices.size()*m_num_img_srcs);
  std::pair<std::vector<std::string>, int> sample;
  size_t cur_offset = 0;
  data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);

  std::unordered_map<std::string, size_t> names;
  size_t jj = 0;
  for (size_t j=0; j<m_my_global_indices.size(); j++) {
    size_t base_index = m_my_global_indices[j];
    sample = reader->get_sample(base_index);
    for (size_t k=0; k<sample.first.size(); k++) {
      size_t index = base_index*m_num_img_srcs + k; 
      size_t file_len = 0;
      if (names.find(sample.first[k]) != names.end()) {
        file_len = names[sample.first[k]];
      } else {
        file_len = get_file_size(m_dir, sample.first[k]);
        names[sample.first[k]] = file_len;
      }

      my_file_sizes[jj].global_index = index;
      my_file_sizes[jj].num_bytes = file_len;
      my_file_sizes[jj].offset = cur_offset;
      my_file_sizes[jj].rank = m_rank;
      cur_offset += my_file_sizes[jj].num_bytes;

      if (my_file_sizes[jj].num_bytes == 0) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: " << " j: " << j 
          << " file size is 0";
        throw lbann_exception(err.str());
      }
      ++jj;
    }
  }

  exchange_file_sizes(my_file_sizes, m_num_global_indices*m_num_img_srcs);
}

void data_store_multi_images::read_files() {
  std::stringstream err;
  data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);
  std::pair<std::vector<std::string>, int> sample;
  for (size_t j=0; j<m_my_global_indices.size(); j++) {
    size_t base_index = m_my_global_indices[j];
    sample = reader->get_sample(base_index);
    for (size_t k=0; k<sample.first.size(); k++) {
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

      if (offset + file_len > m_data.size()) {
        err << __FILE__ << " " << __LINE__ << " :: " << " j: " << j 
          << " of " << m_my_global_indices.size() << " offset: " << offset
          << " file_len: " << file_len << " offset+file_len: "
          << offset+file_len << " m_data.size(): " << m_data.size()
          << "\noffset+file_len must be <= m_data.size()";
        throw lbann_exception(err.str());
      }  

      load_file(m_dir, sample.first[k], &m_data[offset], file_len);
    }
  }
}

void data_store_multi_images::test_data() {
  std::cerr << m_rank << " :: STARTING data_store_multi_images::test_data()\n";
  data_reader_multi_images *reader = dynamic_cast<data_reader_multi_images*>(m_reader);

  std::vector<unsigned char> b;
  std::pair<std::vector<std::string>, int> sample;
  size_t j = -1;
  for (auto t : m_my_minibatch_indices) {
    int idx = (*m_shuffled_indices)[t];
    sample = reader->get_sample(idx);
    for (size_t k=0; k<sample.first.size(); k++) {
      ++j;
      size_t index = idx*m_num_img_srcs+k;

      std::string imagepath = m_dir + sample.first[k];
      std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
      if (! in.good()) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "failed to open " << imagepath << " for reading";
        throw lbann_exception(err.str());
      }

      in.seekg(0, std::ios::end);
      size_t sz = in.tellg();
      in.seekg(0, std::ios::beg);
      b.resize(sz);
      in.read((char*)&b[0], sz);
      in.close();

      //compare to m_my_data
      std::vector<unsigned char> *v;
      get_data_buf(idx, v, 0, (int)k); 
      if (b != *v) {
        std::stringstream err;
        err << ">>>>>>>>>>>>>> "
            << "data_store_multi_images::test_data; error: b != *v; b.size: " 
            << b.size() << "  v.size: " << v->size() << "; " << j << " of " 
            << m_my_minibatch_indices.size() << " file_sizes[index]: " 
            << m_file_sizes[index] << "\nindex: " << index
            << "\nfn:           " << sample.first[k] 
            << "\nindex_2_name: " << idx_2_name[index]
            << "\nname_2_size:  " << name_2_size[sample.first[k]]
            << "\n\n";
        throw lbann_exception(err.str());
      } 
    } //for (size_t k=0; k<sample.first.size()
  } //for (auto t : m_my_minibatch_indices
  std::cerr << "rank: " << m_rank << " role: " << m_reader->get_role() << " :: data_store_imagenet::test_data: PASSES!\n";
}

}  // namespace lbann
