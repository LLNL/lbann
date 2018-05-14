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

#include "lbann/data_store/data_store_csv.hpp"
#include "lbann/data_readers/data_reader_csv.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include <unordered_set>

namespace lbann {

data_store_csv::data_store_csv(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m) {
  set_name("data_store_csv");
}

data_store_csv::~data_store_csv() {
}

void data_store_csv::setup() {
  double tm1 = get_time();
  std::stringstream err;

  if (m_master) {
    std::cerr << "starting data_store_csv::setup() for role: " 
              << m_reader->get_role() << "\n"
              << "calling generic_data_store::setup()\n";
  }
  generic_data_store::setup();

  if (! m_in_memory) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    //sanity check
    csv_reader *reader = dynamic_cast<csv_reader*>(m_reader);
    if (reader == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<data_reader_csv*>(m_reader) failed";
      throw lbann_exception(err.str());
    }
    m_csv_reader = reader;

    if (m_np != reader->get_num_parallel_readers() && ! is_subsidiary_store()) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "num_parallel_readers(): " << reader->get_num_parallel_readers() 
          << " m_np: " << m_np 
          << "; for this data_store num_readers must be the same as procs per model;\n"
          << " if this isn't acceptable, please notify Dave Hysom so he can fix.\n"
          << "reader role: " << m_reader->get_role();
      throw lbann_exception(err.str());
    }

    if (m_master) {
      std::vector<DataType> v = reader->fetch_line_label_response(0);
      m_vector_size = v.size();
    }
    m_comm->world_broadcast<int>(0, m_vector_size);

    if (is_subsidiary_store()) {
      return;
    }

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
    std::cerr << "TIME for data_store_csv setup: " << get_time() - tm1 << "\n";
  }
}

void data_store_csv::get_data_buf_DataType(int data_id, std::vector<DataType> *&buf) {
static int n = 0;
  if (m_my_minibatch_data.find(data_id) == m_my_minibatch_data.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find data_id: " << data_id << " in m_my_minibatch_data\n"
        << "m_my_minibatch_data.size(): " << m_my_minibatch_data.size() << "\n"
        << "role: " << m_reader->get_role() << "  n: " << n;
    throw lbann_exception(err.str());
  }
  n += 1;
  buf = &m_my_minibatch_data[data_id];
}

void data_store_csv::get_my_indices(std::unordered_set<int> &indices, int p) {
  indices.clear();
  std::vector<int> &v = m_all_minibatch_indices[p];
  for (auto t : v) {
    int index = (*m_shuffled_indices)[t];
    if (m_data.find(index) != m_data.end()) {
      indices.insert(index);
    }
  }
}

void data_store_csv::get_indices(std::unordered_set<int> &indices, int p) {
  indices.clear();
  std::vector<int> &v = m_all_minibatch_indices[p];
  for (auto t : v) {
    indices.insert((*m_shuffled_indices)[t]);
  }
}


void data_store_csv::exchange_data() {
  double tm1 = get_time();
  std::stringstream err;

  //get indices I need for the next epoch, and start receives
  std::unordered_set<int> indices;
  get_indices(indices, m_rank);
  std::vector<El::mpi::Request<DataType>> recv_req(indices.size());

  m_my_minibatch_data.clear();
  size_t jj = 0;
  for (auto data_id : indices) {
    m_my_minibatch_data[data_id].resize(m_vector_size);
    int owner = get_index_owner(data_id);
    if (owner >= m_np or owner < 0) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << " ERROR: bad rank for owner in nb_recv; owner: " << owner << " data_id: " << data_id << " jj: " << jj+1 << " of " << indices.size();
      throw lbann_exception(err.str());
    }
    m_comm->nb_tagged_recv<DataType>(m_my_minibatch_data[data_id].data(), m_vector_size, owner, data_id, recv_req[jj++], m_comm->get_model_comm());
  }

  //start sends to all processors
  std::vector<std::vector<El::mpi::Request<DataType>>> send_req(m_np);
  for (int p=0; p<m_np; p++) {
    get_my_indices(indices, p);
    send_req[p].resize(indices.size());
    jj = 0;
    for (auto data_id : indices) {
      if (m_data.find(data_id) == m_data.end()) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << " m_data.find(" << data_id << ") failed.";
        throw lbann_exception(err.str());
      }
      if (m_data[data_id].size() != (size_t)m_vector_size) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << " m_data[" << data_id << "].size = " << m_data[data_id].size()
            << " should be: " << m_vector_size << "; " << jj+1
            << " of " << indices.size()
            << " m_reader->get_role: " << m_reader->get_role();
        throw lbann_exception(err.str());
      }
      m_comm->nb_tagged_send<DataType>(m_data[data_id].data(), m_vector_size, p, data_id, send_req[p][jj++], m_comm->get_model_comm());
    }
  }

  //wait for sends to finish
  if (m_master) {
    for (size_t i=0; i<send_req.size(); i++) {
      m_comm->wait_all(send_req[i]);
    }
  }

  //wait for recvs to finish
  m_comm->wait_all(recv_req);

  if (m_master) {
    std::cerr << "TIME for data_store_csv::exchange_data(): " 
             << get_time() - tm1 << "; role: " << m_reader->get_role() << "\n";
  }
}

void data_store_csv::populate_datastore() {
  for (auto idx : m_my_datastore_indices) {
    m_data[idx] = m_csv_reader->fetch_line_label_response(idx);
    if (m_data[idx].size() != (size_t) m_vector_size) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "m_data[" << idx << "].size() is " << m_data[idx].size()
          << " but should be: " << m_vector_size
          << "; m_data.size: " << m_data.size() << "\n";
      throw lbann_exception(err.str());
    }
  }
}

}  // namespace lbann
