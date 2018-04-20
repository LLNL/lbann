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

#include "lbann/data_store/data_store_imagenet.hpp"
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

void data_store_imagenet::setup() {
  double tm1 = get_time();
  if (m_rank == 0) {
    std::cerr << "starting data_store_imagenet::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }

  set_name("data_store_imagenet");

  //sanity check
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  if (reader == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "dynamic_cast<image_data_reader*>(m_reader) failed";
    throw lbann_exception(err.str());
  }


  //optionally run some tests at the end of setup()
  bool run_tests = false;
  if (options::get()->has_bool("test_data_store") && options::get()->get_bool("test_data_store")) {
    run_tests = true;
    options::get()->set_option("exit_after_setup", true);
  }

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    data_store_image::setup();

    if (run_tests) {
      test_file_sizes();
      test_data();
    }  

    double tm2 = get_time();
    if (m_rank == 0) {
      std::cerr << "data_store_imagenet setup time: " << tm2 - tm1 << std::endl;
    }
  }
}


void data_store_imagenet::test_data() {
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  std::vector<unsigned char> b;
  std::vector<unsigned char> *datastore_buf;
  for (auto t : m_my_minibatch_indices_v) {
    int idx = (*m_shuffled_indices)[t];

    //read directly from file
    std::string imagepath = m_dir + image_list[idx].first;
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

    //get from datastore
    get_data_buf(idx, datastore_buf, 0);
    if (b != *datastore_buf) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " :: data_store_imagenet::test_data, b != v; b.size: " 
          << b.size() << "  datstore_buf->size: " << datastore_buf->size();
      throw lbann_exception(err.str());
    } 
  }

  std::cerr << "rank: " << m_rank << " role: " << m_reader->get_role() << " :: data_store_imagenet::test_data: PASSES!\n";
}

void data_store_imagenet::test_file_sizes() {
  if (m_master) {
    std::cerr << m_rank << " :: STARTING data_store_imagenet::test_file_sizes()\n";
  }
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  for (auto t : m_file_sizes) {
    size_t len = get_file_size(m_dir, image_list[t.first].first);
    if (t.second != len || len == 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "m_file_sizes[" << t.first << "] = " << t.second
          << " but actual size appears to be " << len;
      throw lbann_exception(err.str());
    }
  }
  std::cerr << "rank:  " << m_rank << " role: " << m_reader->get_role() << " :: data_store_imagenet::test_file_sizes: PASSES!\n";
}


void data_store_imagenet::read_files() {
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  size_t j = 0;
  for (auto index : m_my_datastore_indices) {
    if (m_offsets.find(index) == m_offsets.end()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
          << " m_offsets.find(index) failed for index: " << index;
      throw lbann_exception(err.str());
    }
    size_t offset = m_offsets[index];
    if (m_file_sizes.find(index) == m_file_sizes.end()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
          << " m_file_sizes.find(index) failed for index: " << index;
      throw lbann_exception(err.str());
    }
    size_t file_len = m_file_sizes[index];
    if (offset + file_len > m_data.size()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " << " j: " << j 
        << " of " << m_my_minibatch_indices_v.size() << " offset: " << offset
        << " file_len: " << file_len << " offset+file_len: "
        << offset+file_len << " m_data.size(): " << m_data.size()
        << "\noffset+file_len must be <= m_data.size()";
      throw lbann_exception(err.str());
    }
    load_file(m_dir, image_list[index].first, &m_data[offset], file_len);
    ++j;
  }
}

void data_store_imagenet::get_file_sizes() {
  if (m_master) std::cerr << "starting data_store_imagenet::get_file_sizes\n";
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();

  std::vector<int> global_indices(m_my_datastore_indices.size());
  std::vector<int> bytes(m_my_datastore_indices.size());
  std::vector<size_t> offsets(m_my_datastore_indices.size());

  size_t cur_offset = 0;
  size_t j = 0;
  for (auto index : m_my_datastore_indices) {
    global_indices[j] = index;
    bytes[j] = get_file_size(m_dir, image_list[index].first);
    offsets[j] = cur_offset;
    cur_offset += bytes[j];
    ++j;
  }

  exchange_file_sizes(global_indices, bytes, offsets, m_num_global_indices);
}


}  // namespace lbann
