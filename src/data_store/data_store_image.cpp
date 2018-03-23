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

#include "lbann/data_store/data_store_image.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

data_store_image::~data_store_image() {
  MPI_Win_free( &m_win );
}

void data_store_image::setup() {

  if (m_master) std::cerr << "starting data_store_image::setup(); calling generic_data_store::setup()\n";
  generic_data_store::setup();

  //@todo needs to be designed and implemented!
  if (! m_in_memory) {
    LBANN_ERROR("not yet implemented");
  } 
  
  else {
    get_minibatch_index_vector();
    //m_my_minibatch_data.resize(m_my_minibatch_indices_v.size()*m_num_img_srcs);

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "calling get_file_sizes\n";
    get_file_sizes();

    if (m_master) std::cerr << "calling allocate_memory\n";
    allocate_memory();

    if (m_master) std::cerr << "calling read_files\n";
    read_files();
    if (m_master) std::cerr << "DONE! calling read_files\n";

    MPI_Win_create((void*)m_data.data(), m_data.size(), 1, MPI_INFO_NULL, m_comm->get_model_comm().comm, &m_win);

    if (m_master) std::cerr << "calling exchange_data\n";
    exchange_data();

    if (m_extended_testing) {
      if (m_master) std::cerr << "calling extended_testing\n";
      extended_testing();
    }
  }
}

void data_store_image::extended_testing() {
#if 0
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
#endif
}

void data_store_image::get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx) {
  std::stringstream err;
  int index = data_id * m_num_img_srcs + multi_idx;
  if (m_my_minibatch_data.find(index) == m_my_minibatch_data.end()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find index: " << index << " m_num_img_srcs: " << m_num_img_srcs
        << " multi_idx: " << multi_idx << " in m_my_data_hash; role: "
        << m_reader->get_role();
    throw lbann_exception(err.str());
  }

/*
  if (m_extended_testing) {
    static bool first_extended = true;
    if (first_extended && m_master) {
      std::cerr << "\ndata_store_image::get_data_buf -- running extended_testing\n\n";
      first_extended = false;
    }
    if (m_file_sizes.find(index) == m_file_sizes.end()) {
      err << __FILE__ << " " << __LINE__ << " :: "
        << "m_file_sizes has no key for index: " << index;
      throw lbann_exception(err.str());
    }
    if (m_test_filenames.find(index) == m_test_filenames.end()) {
      err << __FILE__ << " " << __LINE__ << " :: "
        << "m_test_filenames has no key for index: " << index;
      throw lbann_exception(err.str());
    }
    std::vector<unsigned char> testme(m_file_sizes[index]);
    std::string dir("");
    load_file(dir, m_test_filenames[index], &testme[0], testme.size());
    if (testme != m_my_minibatch_data[idx]) {
      err << __FILE__ << " " << __LINE__ << " :: "
        << m_rank << " test failed for idx: " << idx;
      throw lbann_exception(err.str());
    }
  } //end: if (m_extended_testing) {
*/

  buf = &m_my_minibatch_data[index];
}

void data_store_image::allocate_memory() {
  size_t m = 0;
  for (auto t : m_my_datastore_indices) {
    for (size_t i=0; i<m_num_img_srcs; i++) {
      m += m_file_sizes[t*m_num_img_srcs+i];
    }    
  }    
  m_data.resize(m);
}

void data_store_image::load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz) {
  std::string imagepath;
  if (dir != "") {
    imagepath = dir + fn;
  } else {
    imagepath = fn;
  }
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (!in) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << imagepath << " for reading"
        << "; dir: " << dir << "  fn: " << fn;
    throw lbann_exception(err.str());
  }
  in.read((char*)p, sz);
  if ((int)sz != in.gcount()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to read " << sz << " bytes from " << imagepath
        << " num bytes read: " << in.gcount();
    throw lbann_exception(err.str());
  }
  in.close();
}

void data_store_image::exchange_data() {
  std::stringstream err;
  double tm1 = get_time();
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  m_my_minibatch_data.clear();

  for (auto idx : m_my_minibatch_indices_v) {
    size_t base_index = (*m_shuffled_indices)[idx];
    for (size_t i=0; i<m_num_img_srcs; i++) {
      int index = base_index*m_num_img_srcs+i;

      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        err << __FILE__ << " :: " << __LINE__ << " :: "
            << index << " is not a key in m_file_sizes";
        throw lbann_exception(err.str());
      }

      if (m_offsets.find(index) == m_offsets.end()) {
        err << __FILE__  << " :: " << __LINE__ << " :: "
          << " m_offsets.find(index) failed";
        throw lbann_exception(err.str());
      }

      int file_len = m_file_sizes[index];
      int owner = get_index_owner(index);
      int offset = m_offsets[index];
      m_my_minibatch_data[index].resize(file_len);
      MPI_Get(m_my_minibatch_data[index].data(), file_len, MPI_BYTE,
              owner, offset, file_len, MPI_BYTE, m_win);
    }
  }

  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  double tm2 = get_time();
  if (m_rank == 0) {
    std::cerr << "data_store_image::exchange_data() time: " << tm2 - tm1 << std::endl;
  }
}

void data_store_image::exchange_file_sizes(std::vector<Triple> &my_file_sizes, int num_global_indices) {
  //exchange files sizes
  std::vector<Triple> global_file_sizes(num_global_indices);

//debug block
for (size_t j=0; j<global_file_sizes.size(); j++) {
  global_file_sizes[j].global_index = -1;
  global_file_sizes[j].num_bytes = -1;
  global_file_sizes[j].offset = 100000;
  global_file_sizes[j].rank = -1;
}

  std::vector<int> d(m_np);
  std::iota(d.begin(), d.end(), 0);
  std::vector<int> d2(m_np, 1);
  std::vector<int> num_bytes(m_np);
  int bytes = my_file_sizes.size()*sizeof(Triple);
  MPI_Allgatherv(&bytes, 1, MPI_INT,
                 num_bytes.data(), d2.data(), d.data(), MPI_INT,
                 m_mpi_comm);

  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + num_bytes[h-1];
  }

/*
  for (size_t j=0; j<m_num_samples.size(); j++) {
    m_num_samples[j] *= sizeof(Triple)*m_num_img_srcs;
  }
*/

if (m_master) {
  std::cerr << "\nnum samples: ";
  for (auto t : num_bytes) std::cerr << t << " ";
  std::cerr << "\ndispl: ";
  for (auto t : disp) std::cerr << t << " ";
  std::cerr << "\n\n";
  std::cerr << "sizeof(Triple): " << sizeof(Triple) << "\n\n";
}

  //@todo: couldn't get m_comm->model_gatherv to work
  //m_comm->model_gatherv(&my_file_sizes[0], my_file_sizes.size(), 
   //                     &global_file_sizes[0], &num_images[0], &disp[0]);
  MPI_Allgatherv(my_file_sizes.data(), my_file_sizes.size()*sizeof(Triple), MPI_BYTE,
                 global_file_sizes.data(), num_bytes.data(), disp.data(), MPI_BYTE,
                 m_mpi_comm);


for (size_t j=0; j<global_file_sizes.size(); j++) {
  if (global_file_sizes[j].global_index == -1 ||
  global_file_sizes[j].num_bytes == -1 ||
  global_file_sizes[j].offset == 1000000  ||
  global_file_sizes[j].rank == -1)
  throw lbann_exception("BBBBBBBBBBBBBBAAAAAAAAAAAAADDDDDDDD");
}

  size_t j = 0;
  for (auto t : global_file_sizes) {
    m_file_sizes[t.global_index] = t.num_bytes;
    m_offsets[t.global_index] = t.offset;
    if (t.num_bytes <= 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: " 
        << "num_bytes  <= 0 (" << t.num_bytes << ")"
        << " for # " << j << " of " << global_file_sizes.size();
      throw lbann_exception(err.str());
    }
    //m_owner_mapping[t.global_index] = t.rank;
    ++j;
  }
}

}  // namespace lbann
