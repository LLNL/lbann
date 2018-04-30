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

#include "lbann/data_store/data_store_pilot2_molecular.hpp"
#include "lbann/data_readers/data_reader_pilot2_molecular.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include <unordered_set>

namespace lbann {

data_store_pilot2_molecular::data_store_pilot2_molecular(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m) {
  set_name("data_store_pilot2_molecular");
}

data_store_pilot2_molecular::~data_store_pilot2_molecular() {
}

void data_store_pilot2_molecular::setup() {
  double tm1 = get_time();
  std::stringstream err;
  m_owner = (int)m_reader->get_compound_rank() == (int)m_rank;
  m_owner_rank = m_reader->get_compound_rank();

  if (m_owner) std::cerr << "starting data_store_pilot2_molecular::setup() for role: " 
          << m_reader->get_role() << "; owning processor: " << m_owner_rank << std::endl; 
  if (m_owner) std::cerr << "calling generic_data_store::setup()\n";
  generic_data_store::setup();

  if (! m_in_memory) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    //sanity check
    pilot2_molecular_reader *reader = dynamic_cast<pilot2_molecular_reader*>(m_reader);
    if (reader == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<data_reader_pilot2_molecular*>(m_reader) failed";
      throw lbann_exception(err.str());
    }
    m_pilot2_reader = reader;

    // get list of indices used in calls to generic_data_reader::fetch_data 
    // for this processor
    get_minibatch_index_vector();

    // get list of indices used in calls to generic_data_reader::fetch_data 
    // for all processors
    if (m_master) std::cerr << "calling exchange_mb_indices\n";
    exchange_mb_indices();

    // allocate storage for the data that will be passed to the data reader's
    // fetch_datum method. 
    m_data_buffer.resize(omp_get_max_threads());
    int num_features = m_pilot2_reader->get_num_features();
    int num_neighbors = m_pilot2_reader->get_num_neighbors();
    for (size_t j=0; j<m_data_buffer.size(); j++) {
      m_data_buffer[j].resize(num_features * (num_neighbors+1));
    }

    if (m_owner) {
      std::cerr << "calling construct_data_store()\n";
      construct_data_store();
    }

    if (m_owner) std::cerr << "calling build_nabor_map()\n";
    build_nabor_map();

    if (m_owner) std::cerr << "calling exchange_data()\n";
    exchange_data();
  }

  if (m_owner) {
    std::cerr << "data_store_pilot2_molecular::setup time: " << get_time() - tm1 << "\n";
  }
}

void data_store_pilot2_molecular::construct_data_store() {
  std::stringstream err;

  // get the feature and neighbor data from the pilot2_molecular data reader
  if (m_pilot2_reader->get_word_size() == 4) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not implemented for word_size = 4; please ask Dave Hysom to fix";
    throw lbann_exception(err.str());
  }
  double *features_8 = m_pilot2_reader->get_features_8();

  int num_samples_per_frame = m_pilot2_reader->get_num_samples_per_frame();
  
  for (size_t j=0; j<m_num_global_indices; j++) {
    int data_id = (*m_shuffled_indices)[j];
    int num_features = m_pilot2_reader->get_num_features();
    fill_in_data(data_id, num_samples_per_frame, num_features, features_8);
  }
}


// replicated code from data_reader_pilot2_molecular::fetch_molecule
void data_store_pilot2_molecular::fill_in_data(
    const int data_id, 
    const int num_samples_per_frame, 
    const int num_features, 
    double *features) {
  const int frame = m_pilot2_reader->get_frame(data_id);
  const int frame_offset = frame * num_features * num_samples_per_frame;
  const int intra_frame_data_id = data_id - frame * num_samples_per_frame;
  double *data = features + frame_offset + intra_frame_data_id * num_features;
  if (m_data.find(data_id) != m_data.end()) {
    std::stringstream err;
    err << __FILE__  << " :: " << __LINE__ << " :: "
        << " duplicate data_id: " << data_id;
    throw lbann_exception(err.str());
  }
  m_data[data_id].resize(num_features);
  for (int i=0; i<num_features; i++) {
    m_data[data_id][i] = m_pilot2_reader->scale_data<double>(i, data[i]);
  }
}




void data_store_pilot2_molecular::build_nabor_map() {
  //bcast neighbor data
  size_t sz;
  if (m_owner) {
    sz = m_pilot2_reader->get_neighbors_data_size();
  }
  m_comm->world_broadcast<size_t>(0, sz);

  double *neighbors_8;
  std::vector<double> work;
  if (m_owner) {
    neighbors_8 = m_pilot2_reader->get_neighbors_8();
  } else {
    work.resize(sz);
    neighbors_8 = work.data();
  }
  m_comm->world_broadcast<double>(0, neighbors_8, sz);

  //fill in the nabors map
  for (auto data_id : (*m_shuffled_indices)) {
    int frame = m_pilot2_reader->get_frame(data_id);
    int max_neighborhood = m_pilot2_reader->get_max_neighborhood();
    int num_samples_per_frame = m_pilot2_reader->get_num_samples_per_frame();
    const int neighbor_frame_offset = frame * num_samples_per_frame * (2 * max_neighborhood);
    const int intra_frame_data_id = data_id - frame * num_samples_per_frame;
    int num_neighbors = m_pilot2_reader->get_num_neighbors();
    m_neighbors[data_id].reserve(num_neighbors);
    double *neighbor_data = neighbors_8 + neighbor_frame_offset + intra_frame_data_id * (2 * max_neighborhood);
    for (int i=1; i<num_neighbors + 1; ++i) {
      int neighbor_id = neighbor_data[i];
      m_neighbors[data_id].push_back(neighbor_id);
    }
  }
}

void data_store_pilot2_molecular::get_data_buf(int data_id, int tid, std::vector<double> *&buf) {
  std::stringstream err;
  std::vector<double> &v = m_data_buffer[tid];
  std::fill(v.begin(), v.end(), 0.0);
  if (m_neighbors.find(data_id) == m_neighbors.end()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << data_id << " not found in m_neighbors (primary molecule)";
    throw lbann_exception(err.str());
  }
  if (m_my_molecules.find(data_id) == m_my_molecules.end()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << data_id << " not found in m_my_molecules";
    throw lbann_exception(err.str());
  }

  //fill in data for the primary molecule
  size_t jj = 0;
  std::vector<double> &d1 = m_my_molecules[data_id];
  for (size_t j=0; j<d1.size(); j++) {
    v[jj++] = d1[j];
  }

  //fill in data for the primary molecule's neighbor
  std::vector<int> &nabors = m_neighbors[data_id];
  int num_features = m_pilot2_reader->get_num_features();
  for (size_t h=0; h<nabors.size(); h++) {
    if (nabors[h] != -1) {
      if (m_my_molecules.find(data_id) == m_my_molecules.end()) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << nabors[h] << " not found in m_my_molecules (neighbor)";
        throw lbann_exception(err.str());
      }
      std::vector<double> &d2 = m_my_molecules[nabors[h]];
      for (size_t i=0; i<d2.size(); i++) {
        v[jj++] = d2[i];
      }
    } else {
      jj += num_features;
    }
  }
  buf = &v;
}

void data_store_pilot2_molecular::get_required_molecules(std::unordered_set<int> &required_molecules, int p) {
  required_molecules.clear();
  std::vector<int> &v = m_all_minibatch_indices[p];
  for (auto t : v) {
    int data_id = (*m_shuffled_indices)[t];
    required_molecules.insert(data_id);
    if (m_neighbors.find(data_id) == m_neighbors.end()) {
      std::stringstream err;
      err << __FILE__  << " :: " << __LINE__ << " :: "
          << " m_neighbors.find(" << data_id << " failed";
      throw lbann_exception(err.str());
    }
    for (auto t2 : m_neighbors[data_id]) {
      if (t2 != -1) {
        required_molecules.insert(t2);
      }
    }
  }
}

void data_store_pilot2_molecular::exchange_data() {
  double tm1 = get_time();
  std::stringstream err;

  //get set of molecules required for the next epoch for myself
  std::unordered_set<int> required_molecules;
  get_required_molecules(required_molecules, m_rank);

  //start receives for my required molecules
  m_my_molecules.clear();
  int num_features = m_pilot2_reader->get_num_features();

  std::vector<El::mpi::Request<double>> recv_req(required_molecules.size());
  size_t jj = 0;
  for (auto data_id : required_molecules) {
    m_my_molecules[data_id].resize(num_features);
    m_comm->nb_tagged_recv<double>(
          m_my_molecules[data_id].data(), num_features, m_owner_rank, 
          data_id, recv_req[jj++], m_comm->get_world_comm());
  }

  //owner starts sends
  std::vector<std::vector<El::mpi::Request<double>>> send_req;
  if (m_owner) {
    send_req.resize(m_np);
    for (int p = 0; p<m_np; p++) {
      jj = 0;
      get_required_molecules(required_molecules, p);
      send_req[p].resize(required_molecules.size());
      for (auto data_id : required_molecules) {
        m_comm->nb_tagged_send<double>(
           m_data[data_id].data(), num_features, p, 
           data_id, send_req[p][jj++], m_comm->get_world_comm());
      }
    }
  }

  //wait for sends to finish
  if (m_owner) {
    for (size_t i=0; i<send_req.size(); i++) {
      m_comm->wait_all<double>(send_req[i]);
    }
  }

  //wait for recvs to finish
  m_comm->wait_all<double>(recv_req);

  if (m_owner) {
    std::cout << "role: " << m_reader->get_role() << " data_store_pilot2_molecular::exchange_data() time: " << get_time() - tm1 << std::endl;
  }
}

}  // namespace lbann
