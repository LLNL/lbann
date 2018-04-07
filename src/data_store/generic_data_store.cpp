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

#include "lbann/data_store/generic_data_store.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/models/model.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <numeric>

namespace lbann {

generic_data_store::generic_data_store(generic_data_reader *reader, model *m) :
    m_reader(reader), 
    m_comm(m->get_comm()),
    m_epoch(0),
    m_in_memory(true),
    m_master(m_comm->am_world_master()), 
    m_model(m),
    m_dir(m_reader->get_file_dir()),
    m_extended_testing(false),
    m_collect_minibatch_indices(true),
    m_mpi_comm(m_comm->get_model_comm().comm)
{
  if (m_comm == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << " m_reader->get_comm is nullptr";
        throw lbann_exception(err.str());
  }
  m_rank = m_comm->get_rank_in_model();
  m_np = m_comm->get_procs_per_model();
  set_name("generic_data_store");
  options *opts = options::get();
  if (m_master) std::cerr << "generic_data_store::generic_data_store; np: " << m_np << "\n";
    if (opts->has_bool("extended_testing") && opts->get_bool("extended_testing")) {
      m_extended_testing = true;
    }
}

void generic_data_store::get_minibatch_index_vector() {
  size_t s2 = 0;
  for (auto t1 : (*m_my_minibatch_indices)) {
    s2 += t1.size();
  }
  m_my_minibatch_indices_v.reserve(s2);
  for (auto t1 : (*m_my_minibatch_indices)) {
    for (auto t2 : t1) {
      m_my_minibatch_indices_v.push_back(t2);
    }
  }
}

void generic_data_store::get_my_datastore_indices() {
  m_num_samples.resize(m_np, 0);
  std::unordered_set<int> mine;
  for (size_t j=0; j<m_shuffled_indices->size(); ++j) {
    int idx = (*m_shuffled_indices)[j];
    int owner = idx % m_np;
    m_num_samples[owner] += 1;
    if (owner == m_rank) {
      m_my_datastore_indices.insert(idx);
    }
  }
}

void generic_data_store::setup() {
  set_shuffled_indices( &(m_reader->get_shuffled_indices()) );
  set_num_global_indices();
  m_num_readers = m_reader->get_num_parallel_readers();
  if (m_master) {
    std::cerr << "data_reader type is: " << m_reader->get_type() << "\n";
  }

  // get the set of global indices used by this processor in
  // generic_data_reader::fetch_data(). Note that these are
  // "original' indices, not shuffled indices, i.e, these indices
  // remain constant through all epochs
  if (m_collect_minibatch_indices) {
    if (m_master) { std::cerr << "calling m_model->collect_indices\n"; }
    m_reader->set_save_minibatch_entries(true);
    if (m_reader->get_role() == "train") {
      m_model->collect_indices(execution_mode::training);
    } else if (m_reader->get_role() == "validate") {
      m_model->collect_indices(execution_mode::validation);
    } else if (m_reader->get_role() == "test") {
      m_model->collect_indices(execution_mode::testing);
    } else {
      std::stringstream s2;
      s2 << __FILE__ << " " << __LINE__ << " :: "
         << " bad role; should be train, test, or validate;"
         << " we got: " << m_reader->get_role();
        throw lbann_exception(s2.str());
    }
    m_reader->set_save_minibatch_entries(false);
  }
  m_my_minibatch_indices = &(m_reader->get_minibatch_indices());
}


size_t generic_data_store::get_file_size(std::string dir, std::string fn) {
  std::string imagepath;
  if (m_dir == "") {
    imagepath = fn;
  } else {
    imagepath = dir + fn;
  }
  struct stat st;
  if (stat(imagepath.c_str(), &st) != 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "stat failed for dir: " << dir
        << " and fn: " << fn;
    throw lbann_exception(err.str());
  }
  return st.st_size;   
}

void generic_data_store::set_shuffled_indices(const std::vector<int> *indices) {
  m_shuffled_indices = indices;
  ++m_epoch;
  if (m_epoch > 1) {
    exchange_data();
  }
  /*
  if (m_rank == 0) {
    bool is_shuffled = false;
    for (size_t j=1; j<indices->size(); j++) {
      if ((*indices)[j-1]+1 != (*indices)[j]) {
        is_shuffled = true;
        break;
      }
    }
    //std::cerr << "IS_SHUFFLED: " << is_shuffled << "\n";
  }
  */
}

void generic_data_store::exchange_mb_counts() {
  int my_num_indices = m_my_minibatch_indices_v.size();
  m_mb_counts.resize(m_np);
  std::vector<int> num(m_np, 1); //num elements to be received from P_j
  MPI_Allgather(&my_num_indices, 1, MPI_INT,
                 m_mb_counts.data(), 1, MPI_INT, m_mpi_comm);
}

void generic_data_store::exchange_mb_indices() {
  exchange_mb_counts();
  //setup data structures to exchange minibatch indices with all processors
  //displacement vector
  std::vector<int> displ(m_np);
  displ[0] = 0;
  for (size_t j=1; j<m_mb_counts.size(); j++) {
    displ[j] = displ[j-1] + m_mb_counts[j-1];
  }

  //recv vector
  int n = std::accumulate(m_mb_counts.begin(), m_mb_counts.end(), 0);
  std::vector<int> all_indices(n);

  //receive the indices
  MPI_Allgatherv(
    m_my_minibatch_indices_v.data(), m_my_minibatch_indices_v.size(), MPI_INT, 
    all_indices.data(), m_mb_counts.data(), displ.data(),
    MPI_INT, m_mpi_comm);

  //fill in the final data structure
  m_all_minibatch_indices.resize(m_np);
  for (int j=0; j<m_np; j++) {
    m_all_minibatch_indices[j].reserve(m_mb_counts[j]);
    for (int i=displ[j]; i<displ[j]+m_mb_counts[j]; i++) {
      m_all_minibatch_indices[j].push_back(all_indices[i]);
    }
  }
}


}  // namespace lbann
