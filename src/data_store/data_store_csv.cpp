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
#include <set>

using namespace std;


namespace lbann {

data_store_csv::data_store_csv(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m) {
  set_name("data_store_csv");
}

data_store_csv::~data_store_csv() {
  if (!m_use_two_sided_comms) {
    MPI_Win_free( &m_win );
  }  
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

    if (m_np != reader->get_num_parallel_readers()) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "num_parallel_readers(): " << reader->get_num_parallel_readers() 
          << " m_np: " << m_np 
          << "; for this data_store num_readers must be the same as procs per model;\n"
          << " if this isn't acceptable, please notify Dave Hysom so he can fix.\n";
      throw lbann_exception(err.str());
    }

    if (m_master) std::cerr << "calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();

    if (m_use_two_sided_comms) {
      if (m_master) std::cerr << "calling get_all_minibatch_indices()\n";
      exchange_mb_indices();
    }  

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();
    
    if (m_master) {
      std::cerr << "calling: reader->fetch_line_label_response(" << (*m_shuffled_indices)[0] << ");\n";
      std::vector<DataType> v = reader->fetch_line_label_response((*m_shuffled_indices)[0]);
      m_vector_size = v.size();
    }
    MPI_Bcast(&m_vector_size, 1, MPI_INT, 0, m_mpi_comm);

    if (m_master) std::cerr << "calling populate_datastore()\n";
    if (m_use_two_sided_comms) {
      populate_datastore_two_sided(); 
    } else {
      populate_datastore(); 
    }

    if (m_master) std::cerr << "calling exchange_offsets()\n";
    exchange_offsets();

#if 0
    if (m_extended_testing) {
      static bool first = true;
      if (m_master && first) {
        std::cerr << "\nrunning extended testing (after exchange_offsets)\n\n";
        first = false;
      }
      for (auto data_id : m_my_datastore_indices) {
        if (m_offset_mapping.find(data_id) == m_offset_mapping.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << m_rank << "  m_offset_mapping.find(" << data_id << ") failed";
          throw lbann_exception(err.str());
        }
      }
    }
#endif

    //allocate buffers to be passed to fetch_datum
    m_my_minibatch_data.resize(m_my_minibatch_indices_v.size());
    for (size_t i=0; i< m_my_minibatch_data.size(); i++) {
      m_my_minibatch_data[i].resize(m_vector_size);
    }

    if (! m_use_two_sided_comms) {
      MPI_Win_create(m_data.data(), m_data.size()*sizeof(DataType), sizeof(DataType), MPI_INFO_NULL, m_mpi_comm, &m_win);
    } 

    else {
      if (m_master) std::cerr << "data_store_csv::calling exchange_mb_counts\n";
      exchange_mb_indices();
    }

    if (m_master) std::cerr << "calling exchange_data()\n";
    exchange_data();
  }

  if (m_master) {
    std::cerr << "data_store_csv::setup time: " << get_time() - tm1 << "\n";
  }
}

void data_store_csv::get_data_buf_DataType(int data_id, std::vector<DataType> *&buf) {
  std::stringstream err;
  if (m_my_data_hash.find(data_id) == m_my_data_hash.end()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find data_id: " << data_id << " in m_my_data_hash";
    throw lbann_exception(err.str());
  }
  int idx = m_my_data_hash[data_id];
  if (m_extended_testing) {
    static bool first = true;
    if (m_master && first) {
      std::cerr << "\nrunning extended testing (in get_data_buf_DataType)\n\n";
      first = false;
    }
    std::vector<DataType> v = m_csv_reader->fetch_line_label_response(data_id);
    if (v != m_my_minibatch_data[idx]) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << " me: " << m_rank << " extended testing failed for data_id "
          << data_id;
      throw lbann_exception(err.str());
    }
  }

  buf = &m_my_minibatch_data[idx];
}

void data_store_csv::exchange_shuffled_indices(
    std::vector<std::unordered_set<int>> &indices, 
    std::vector<int> &indices_send_counts, 
    std::vector<int> &indices_recv_counts) {
  indices.resize(m_np);

  if (m_master) std::cerr << "starting data_store_csv::collect_and_exchange_send_indices\n";

  //get the sets of indices I need to send to other procs
  for (int j=0; j<m_np; j++) {
    std::vector<int> &v = m_all_minibatch_indices[j];
    for (auto t : v) {
      int idx = (*m_shuffled_indices)[t];
      if (m_my_datastore_indices.find(idx) != m_my_datastore_indices.end()) {
        indices[j].insert(idx);
      } 
    }
  }

  indices_send_counts.resize(m_np);
  indices_recv_counts.resize(m_np);
  for (size_t j=0; j<indices.size(); j++) {
    indices_send_counts[j] = indices[j].size();
  }

  MPI_Alltoall(indices_send_counts.data(), 1, MPI_INT,
               indices_recv_counts.data(), 1, MPI_INT, m_mpi_comm);
  if (m_rank == 1) {
    std::cerr << "recv counts: ";
    for (auto t : indices_recv_counts) std::cerr << t << " ";
    std::cerr << "\nsend counts: ";
    for (auto t : indices_send_counts) std::cerr << t << " ";
    std::cerr << "\n\n";
  }
}

void data_store_csv::start_sends_and_recvs() {
}


void data_store_csv::exchange_data_two_sided() {
#if 0
  std::stringstream err;

  compute_send_and_receive_lists();
  std::vector<std::unordered_set<int>> indices;
  std::vector<int> indices_send_counts;
  std::vector<int> indices_recv_counts;
  exchange_shuffled_indices(indices, indices_send_counts, indices_recv_counts);

  //at this point, indices_send_counts[j] contains the # of indices this proc
  //will send to P_j, and recv_counts[j] is the number of indices to receive
  //from P_j.

  std::vector<MPI_Request> reqs_send(m_np);
  std::vector<MPI_Request> reqs_recv(m_np);

  int d_size = sizeof(DataType);
  if (!(d_size == 4 || d_size == 8)) {
    err << __FILE__  << " :: " << __LINE__ << " :: "
        << " unknown or unsupported DataType; size is: " <<sizeof(DataType);
    throw lbann_exception(err.str());
  }

  //setup and start receive buffers
  std::vector<std::vector<DataType>> recv_buffers(m_np);
  for (int j=0; j<m_np; j++) {
    size_t sz = indices_recv_counts[j]*m_vector_size + indices_recv_counts[j] +1;
    recv_buffers[j].resize(sz);
    if (d_size == 4) {
      MPI_Irecv(recv_buffers[j].data(), sz, MPI_FLOAT, j, 0, m_mpi_comm, reqs_recv.data() + j);
    } else {
      MPI_Irecv(recv_buffers[j].data(), sz, MPI_DOUBLE, j, 0, m_mpi_comm, reqs_recv.data() + j);
    }
  }

  //setup and start send buffers
  std::vector<std::vector<DataType>> send_buffers(m_np);
  for (int j=0; j<m_np; j++) {
if (m_rank)std::cerr << "preparing to send to: " << j << "\n";
    const std::unordered_set<int> &sendme = indices[j];
    size_t sz = indices[j].size()*m_vector_size + sendme.size()+1; 
    send_buffers[j].reserve(sz);
    send_buffers[j].push_back(sendme.size());
    for (auto idx : sendme) {
      send_buffers[j].push_back(idx);
      if (m_data.find(idx) == m_data.end()) {
        err << __FILE__  << " :: " << __LINE__ << " :: "
            << " failed to find " << idx << " in m_data";
        throw lbann_exception(err.str());
      }  
      std::vector<DataType> &v = m_data[idx];
      for (auto t : v) {
        send_buffers[j].push_back(t);
      }
    }
    if (send_buffers[j].size() != sz) std::cerr << "\n\nsz/buffer size: " << sz << " / " << send_buffers[j].size() << "\n\n";
    assert(send_buffers[j].size() == sz);
    if (d_size == 4) {
      MPI_Isend(send_buffers[j].data(), send_buffers[j].size(), MPI_FLOAT,
              j, 0, m_mpi_comm, reqs_send.data()+j);
    } else {
      MPI_Isend(send_buffers[j].data(), send_buffers[j].size(), MPI_DOUBLE,
              j, 0, m_mpi_comm, reqs_send.data()+j);
    }
  }

MPI_Barrier(MPI_COMM_WORLD);
exit(0);

#endif
}

void data_store_csv::exchange_data() {
  char b[80];
  sprintf(b, "debug.%d", m_rank);
  std::ofstream out(b);
  if (! out) throw lbann_exception("EROR");
  out << "m_my_minibatch_indices_v.size(): " << m_my_minibatch_indices_v.size() << "\n"
      <<" m_my_minibatch_data.size(); " << m_my_minibatch_data.size() << "\n"
      <<" m_my_minibatch_data[0].size(): " << m_my_minibatch_data[0].size() << "\n"
      << " m_data.size(): " << m_data.size() << "\n";

  if (m_use_two_sided_comms) {
    exchange_data_two_sided();
    return;
  }

  std::stringstream err;
  double tm1 = get_time();
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  m_my_data_hash.clear();

  size_t jj = 0;
  for (auto data_id : m_my_minibatch_indices_v) {
    size_t idx = (*m_shuffled_indices)[data_id];

    if (jj >= m_my_minibatch_data.size()) {
      err << __FILE__  << " :: " << __LINE__ << " :: "
        << " jj: " << jj << " is >= m_my_minibatch_data.size(): "
        << m_my_minibatch_data.size();
      throw lbann_exception(err.str());
    }

    if (m_offset_mapping.find(idx) == m_offset_mapping.end()) {
      err << __FILE__  << " :: " << __LINE__ << " :: "
        << " m_offsets.find(idx) failed";
      throw lbann_exception(err.str());
    }

    m_my_data_hash[idx] = jj;
    int owner = get_index_owner(idx);
    size_t offset = m_offset_mapping[idx];

/*
out << "calling mpi_get; jj: " << jj << " of " << m_my_minibatch_indices_v.size() << "  data_id: " << idx << " owner: " << owner << " offset: " << offset << "  sz: " << m_vector_size*sizeof(DataType) << " m_data: " << m_data.size()
<< " ends_at: " << offset+m_vector_size << " diff: " << (int)m_data.size() - (offset+m_vector_size) << "\n";
out.flush();
*/
    MPI_Get(m_my_minibatch_data[jj].data(), m_vector_size*sizeof(DataType), MPI_BYTE,
              owner, offset, m_vector_size*sizeof(DataType), MPI_BYTE, m_win);
    ++jj;
  }
  MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPUT, m_win);
  double tm2 = get_time();
  if (m_master) {
    std::cout << "data_store_image::exchange_data() time: " << tm2 - tm1 << std::endl;
  }
out.close();
}


void data_store_csv::exchange_offsets() {
  std::vector<Tuple> my_offsets(m_my_datastore_indices.size());
  int jj = 0;
  for (auto t : m_offset_mapping) {
    my_offsets[jj].global_id = t.first;
    my_offsets[jj].offset = t.second;
    jj++;
  }

  std::vector<Tuple> global_offsets(m_num_global_indices);
  std::vector<int> disp(m_num_readers); 
  disp[0] = 0;
  for (int h=1; h<(int)m_num_readers; h++) {
    disp[h] = disp[h-1] + m_num_samples[h-1]*sizeof(Tuple);
  }

  for (size_t j=0; j<m_num_samples.size(); j++) {
    m_num_samples[j] *= sizeof(Tuple);
  }

  //@todo: couldn't get m_comm->model_gatherv to work
  //m_comm->model_gatherv(&my_file_sizes[0], my_file_sizes.size(), 
   //                     &global_file_sizes[0], &num_images[0], &disp[0]);
  MPI_Allgatherv(my_offsets.data(), my_offsets.size()*sizeof(Tuple), MPI_BYTE,
                 global_offsets.data(), &m_num_samples[0], &disp[0], MPI_BYTE,
                 m_mpi_comm);
  size_t j = 0;
  for (auto t : global_offsets) {
    if (m_offset_mapping.find(t.global_id) != m_offset_mapping.end()) {
      assert(m_offset_mapping[t.global_id] == t.offset);
    }
    m_offset_mapping[t.global_id] = t.offset;
    ++j;
  }
}

void data_store_csv::populate_datastore_two_sided() {
  for (auto idx : m_my_datastore_indices) {
    m_data_two_sided[idx] = m_csv_reader->fetch_line_label_response(idx);
  }
}

void data_store_csv::populate_datastore() {
  m_data.resize(m_my_datastore_indices.size() * m_vector_size);
  
  if (m_master) std::cerr << "populating the data store\n";
  size_t jj = 0;
  int j = 0;
  for (auto idx : m_my_datastore_indices) {
    m_offset_mapping[idx] = jj;
    std::vector<DataType> v = m_csv_reader->fetch_line_label_response(idx);
    for (size_t i=0; i<v.size(); i++) {
      m_data[jj++] = v[i];
    }
    if (jj != (1+j)*v.size()) {
      std::cerr << "ERROR: j: " << j << " v.size(): " << v.size() << " jj: " << jj << "\n";
      sleep(3);
    }  
    assert(jj == (1+j)*v.size());
    ++j;
  } 
}

}  // namespace lbann
