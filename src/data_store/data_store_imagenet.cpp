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
#include <sys/stat.h>

namespace lbann {

void data_store_imagenet::get_my_datastore_indices() {
  //compute storage
  int n = 0;
  for (size_t j=0; j<m_num_global_indices; j++) {
    if (j % m_num_readers == m_rank) {
      ++n;
    }
  }
  //get the indices
  m_my_datastore_indices.reserve(n);
  m_my_shuffled_indices.reserve(n);
  for (size_t j=0; j<m_num_global_indices; j++) {
    if (j % m_num_readers == m_rank) {
      m_my_datastore_indices.push_back(j);
      m_my_shuffled_indices.push_back((*m_shuffled_indices)[j]);
    }
  }
}

void data_store_imagenet::compute_owner_mapping() {
  // fill in global index wrt shuffled_indices -> owning processor
  for (size_t j=0; j<m_num_global_indices; j++) {
    int owner = j % m_num_readers;
    m_owner_mapping[(*m_shuffled_indices)[j]] = owner;
  }
}

void data_store_imagenet::compute_my_filenames() {
  m_dir = m_reader->get_file_dir();
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  m_my_files.reserve( m_my_datastore_indices.size() );
  for (size_t j=0; j<m_my_shuffled_indices.size(); j++) {
    m_my_files.push_back(image_list[m_my_shuffled_indices[j]].first);
  }    
}

void data_store_imagenet::compute_num_images() {
  //m_num_images[j] = num images (global indices) owned by P_j
  m_num_images.clear();
  m_num_images.resize(m_num_readers, 0);
  for (size_t j=0; j<m_num_global_indices; j++) {
    m_num_images[j % m_num_readers] += 1;
  }
}

void data_store_imagenet::setup() {
std::stringstream s;
  double tm1 = get_time();
  if (m_rank == 0) {
    std::cerr << "starting data_store_imagenet::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }

  //sanity check
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  if (reader == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "dynamic_cast<imagenet_reader*>(m_reader) failed";
    throw lbann_exception(err.str());
  }


  //optionally run some tests at the end of setup()
  bool run_tests = false;
  if (options::get()->has_bool("test_data_store") && options::get()->get_bool("test_data_store")) {
    run_tests = true;
  }
  
  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
    m_buffers.resize( omp_get_max_threads() );
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
  std::cerr << m_rank << " :: STARTING data_store_imagenet::test_data()\n";
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  std::vector<unsigned char> b;
  std::string file_dir = m_reader->get_file_dir();
  for (auto t : m_my_minibatch_indices) {
    int idx = (*m_shuffled_indices)[t];
    //read directly from file
    std::string imagepath = file_dir + image_list[(*m_shuffled_indices)[t]].first;
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
    if (m_my_data_hash.find(idx) == m_my_data_hash.end()) {
      std::cerr << m_rank << " :: my index: " << idx << " not found in m_my_data_hash\n";
      throw lbann_exception("ERROR #1");
    }
    size_t index = m_my_data_hash[idx];
    std::vector<unsigned char> &v = m_my_minibatch_data[index];
    if (b != v) {
      std::stringstream err;
      err << "ERROR #2; b.size: " << b.size() << "  v.size: " << v.size() << std::endl;
      throw lbann_exception(err.str());
    } 
  }
  std::cerr << "rank: " << m_rank << " role: " << m_reader->get_role() << " :: data_store_imagenet::test_data: PASSES!\n";
}

void data_store_imagenet::test_file_sizes() {
  std::cerr << m_rank << " :: STARTING data_store_imagenet::test_file_sizes()\n";
  imagenet_reader *reader = dynamic_cast<imagenet_reader*>(m_reader);
  const std::vector<std::pair<std::string, int> > & image_list = reader->get_image_list();
  std::string file_dir = m_reader->get_file_dir();
  for (auto t : m_file_sizes) {
    size_t len = get_file_size(file_dir, image_list[t.first].first);
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


void data_store_imagenet::get_file_sizes() {
  //construct a vector of Triples 
  std::string file_dir = m_reader->get_file_dir();
  std::vector<Triple> my_file_sizes(m_my_shuffled_indices.size());
  size_t cur_offset = 0;
  for (size_t j=0; j<m_my_shuffled_indices.size(); j++) {
    size_t index = m_my_shuffled_indices[j];
    my_file_sizes[j].global_index = index;
    my_file_sizes[j].num_bytes = get_file_size(file_dir, m_my_files[j]);
    my_file_sizes[j].offset = cur_offset;
    cur_offset += my_file_sizes[j].num_bytes;
    if (my_file_sizes[j].num_bytes == 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " << " j: " << j 
        << " file size is 0 (" << file_dir << "/" + m_my_files[index];
      throw lbann_exception(err.str());
    }
  }

  //exchange files sizes
  std::vector<Triple> global_file_sizes(m_num_global_indices);
  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + m_num_images[h-1]*sizeof(Triple);
  }

  for (size_t j=0; j<m_num_images.size(); j++) {
    m_num_images[j] *= sizeof(Triple);
  }

  //m_comm->model_gatherv(&my_file_sizes[0], my_file_sizes.size(), 
   //                     &global_file_sizes[0], &num_images[0], &disp[0]);
  MPI_Allgatherv(&my_file_sizes[0], my_file_sizes.size()*sizeof(Triple), MPI_BYTE,
                 &global_file_sizes[0], &m_num_images[0], &disp[0], MPI_BYTE,
                 m_comm->get_model_comm().comm);

  for (auto t : global_file_sizes) {
    m_file_sizes[t.global_index] = t.num_bytes;
    m_offsets[t.global_index] = t.offset;
    if (t.num_bytes <= 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
        << "num_bytes  <= 0 (" << t.num_bytes << ")";
      throw lbann_exception(err.str());
    }
  }
//} catch (const std::logic_error & exc) { std::cerr << m_rank << " :: logic_error thrown: " << exc.what() << std::endl; }
}

}  // namespace lbann
