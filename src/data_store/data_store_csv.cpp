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
      std::cerr << "calling: reader->fetch_line_label_response(" << (*m_shuffled_indices)[0] << ");\n";
      std::vector<DataType> v = reader->fetch_line_label_response(0);
      m_vector_size = v.size();
    }
    MPI_Bcast(&m_vector_size, 1, MPI_INT, 0, m_mpi_comm);

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
    std::cerr << "data_store_csv::setup time: " << get_time() - tm1 << "\n";
  }
}

void data_store_csv::get_data_buf_DataType(int data_id, std::vector<DataType> *&buf) {
  if (m_my_minibatch_data.find(data_id) == m_my_minibatch_data.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find data_id: " << data_id << " in m_my_minibatch_data\n"
        << "m_my_minibatch_data.size(): " << m_my_minibatch_data.size() << "\n"
        << "role: " << m_reader->get_role() ;
    throw lbann_exception(err.str());
  }
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
  std::vector<MPI_Request> recv_req(indices.size());
  std::vector<MPI_Status> recv_status(indices.size());
  m_my_minibatch_data.clear();
  size_t jj = 0;
  for (auto t : indices) {
    m_my_minibatch_data[t].resize(m_vector_size);
    int owner = get_index_owner(t);
    if (owner >= m_np or owner < 0) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << " ERROR: bad rank for owner in MPI_Irecv; owner: " << owner << " index: " << t << " jj: " << jj+1 << " of " << indices.size();
      throw lbann_exception(err.str());
    }
    MPI_Irecv(
      m_my_minibatch_data[t].data(), m_vector_size*sizeof(DataType),  MPI_BYTE,
      owner, t, m_mpi_comm, &(recv_req[jj++]));
  }

  //start sends to all processors
  std::vector<std::vector<MPI_Request>> send_req(m_np);
  std::vector<std::vector<MPI_Status>> send_status(m_np);
  for (int p=0; p<m_np; p++) {
    get_my_indices(indices, p);
    send_req[p].resize(indices.size());
    send_status[p].resize(indices.size());
    jj = 0;
    for (auto t : indices) {
      if (m_data.find(t) == m_data.end()) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << " m_data.find(" << t << ") failed.";
        throw lbann_exception(err.str());
      }
      if (m_data[t].size() != (size_t)m_vector_size) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << " m_data[" << t << "].size = " << m_data[t].size()
            << " should be: " << m_vector_size << "; " << jj+1
            << " of " << indices.size()
            << " m_reader->get_role: " << m_reader->get_role();
        throw lbann_exception(err.str());
      }
      MPI_Isend(
        m_data[t].data(), m_vector_size*sizeof(DataType),  MPI_BYTE,
        p, t, m_mpi_comm, &(send_req[p][jj++]));
    }
  }

  //wait for sends to finish
  if (m_master) {
    for (size_t i=0; i<send_req.size(); i++) {
      MPI_Waitall(send_req[i].size(), send_req[i].data(), send_status[i].data());
    }
  }

  //wait for recvs to finish
  MPI_Waitall(recv_req.size(), recv_req.data(), recv_status.data());

  if (m_master) {
    std::cerr << "role: " << m_reader->get_role() << " data_store_csv::exchange_data() time: " << get_time() - tm1 << std::endl;
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
