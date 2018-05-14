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

  data_store_image::setup();


  if (run_tests && m_in_memory) {
    test_file_sizes();
    test_data();
  }  

  double tm2 = get_time();
  if (m_rank == 0) {
    std::cerr << "TIME for data_store_imagenet setup: " << tm2 - tm1 << std::endl;
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

void data_store_imagenet::read_files(const std::unordered_set<int> &indices) {
  std::stringstream err;
  std::string local_dir = m_reader->get_local_file_dir();
  std::stringstream fp;
  int n = 0;
  double tm = get_time();
  for (auto index : indices) {
    ++n;
    if (n % 100 == 0 && m_master) {
      double time_per_file = (get_time() - tm) / n;
      int remaining_files = indices.size() - n;
      double estimated_remaining_time = time_per_file * remaining_files;
      std::cerr << "P_0, " << m_reader->get_role() << "; read " << n << " of " 
                << indices.size() << " files; elapsed time " << (get_time() - tm)
                << "s; est. remaining time: " << estimated_remaining_time << "\n";
    }
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
    try {
      load_file("", fp.str(), m_data[index].data(), file_len);
    } catch (std::bad_alloc& ba) {
      err << m_rank << " caught std::bad_alloc, what: " << ba.what()
          << " " << getenv("SLURMD_NODENAME") << " file: "
          << fp.str() << " length: " << file_len << "\n";
      throw lbann_exception(err.str()); 
    }
  }
}

void data_store_imagenet::read_files() {
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  for (auto index : m_my_datastore_indices) {
    if (m_file_sizes.find(index) == m_file_sizes.end()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
          << " m_file_sizes.find(index) failed for index: " << index;
      throw lbann_exception(err.str());
    }
    size_t file_len = m_file_sizes[index];
    m_data[index].resize(file_len);
    load_file(m_dir, image_list[index].first, m_data[index].data(), file_len);
  }
}

void data_store_imagenet::get_file_sizes() {
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();

  std::vector<int> global_indices(m_my_datastore_indices.size());
  std::vector<int> bytes(m_my_datastore_indices.size());

  size_t j = 0;
  double tm = get_time();
  for (auto index : m_my_datastore_indices) {
    global_indices[j] = index;
    bytes[j] = get_file_size(m_dir, image_list[index].first);
    ++j;
    if (j % 100 == 0 and m_master) {
      double e = get_time() - tm;
      double time_per_file = e / j;
      int remaining_files = m_my_datastore_indices.size()-j;
      double estimated_remaining_time = time_per_file * remaining_files;
      std::cerr << "P_0: got size for " << j << " of " << m_data_filepaths.size()
                << " files; elapsed time: " << get_time() - tm
                << "s est. remaining time: " << estimated_remaining_time << "s\n";
    }

  }

  exchange_file_sizes(global_indices, bytes);
}

void data_store_imagenet::build_data_filepaths() {
  m_data_filepaths.clear();
  image_data_reader *reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  for (auto index : m_my_datastore_indices) {
    m_data_filepaths[index] = image_list[index].first;
  }
}

}  // namespace lbann
