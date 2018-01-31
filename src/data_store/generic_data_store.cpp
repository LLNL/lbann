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
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/models/model.hpp"

#undef DEBUG
//#define DEBUG

namespace lbann {

generic_data_store::generic_data_store(lbann_comm *comm, generic_data_reader *reader, model *m) :
    m_rank(comm->get_rank_in_model()),
    m_epoch(0),
    m_in_memory(true),
    m_comm(comm), m_master(comm->am_world_master()), m_reader(reader),
    m_model(m)
  {
  /*
    if (options::get()->has_bool("ds_in_memory")) {
      m_in_memory = options::get()->get_bool("ds_in_memory");
    }
    */
  }

void generic_data_store::setup() {

  set_shuffled_indices( &(m_reader->get_shuffled_indices()) );
  m_num_global_indices = m_shuffled_indices->size();
  m_num_readers = m_reader->get_num_parallel_readers();

  if (m_master) {
    std::cerr << "calling m_model->collect_indices\n";
  }
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
  
  //@todo: when we get to not-in-memory mode, probably need to keep
  //       the vector<vector<>> representation. But for all-in-memory,
  //       only need the single m_my_minibatch_indices list.
  const std::vector<std::vector<int> > indices = m_reader->get_minibatch_indices();
  for (auto t1 : indices) {
    for (auto t2 : t1) {
      m_my_minibatch_indices.push_back(t2);
    }
  }

  #ifdef DEBUG
  std::stringstream s;
  s << "debug_" << m_reader->get_role() << "_" << m_rank << ".txt";
  std::ofstream out(s.str().c_str());
  for (auto t : m_my_minibatch_indices) {
    out << t << " ";
  }
  out.close();
  #endif

  get_my_datastore_indices();
}


void generic_data_store::get_my_datastore_indices() {
  for (size_t j=0; j<m_num_global_indices; j++) {
    if (j % m_num_readers == m_rank) {
      m_my_datastore_indices.push_back((*m_shuffled_indices)[j]);
    }
  }    
}

}  // namespace lbann
