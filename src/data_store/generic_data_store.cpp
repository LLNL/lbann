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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

generic_data_store::generic_data_store(lbann_comm *comm, generic_data_reader *reader, model *m) :
    m_rank(comm->get_rank_in_model()),
    m_epoch(0),
    m_in_memory(true),
    m_comm(comm), m_master(comm->am_world_master()), m_reader(reader),
    m_model(m),
    m_dir(m_reader->get_file_dir())
  {
  /*
    if (options::get()->has_bool("ds_in_memory")) {
      m_in_memory = options::get()->get_bool("ds_in_memory");
    }
    */
  }

void generic_data_store::setup(bool test_dynamic_cast, bool run_tests) {
  set_shuffled_indices( &(m_reader->get_shuffled_indices()) );
  set_num_global_indices(); //virtual override in child classes
  m_num_readers = m_reader->get_num_parallel_readers();

  // get the set of global indices used by this processor in
  // generic_data_reader::fetch_data(). Note that these are
  // "original' indices, not shuffled indices, i.e, these indices
  // remain constant through all epochs
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

  //@todo: when we get to not-in-memory mode, probably need to keep
  //       the vector<vector<>> representation. But for all-in-memory,
  //       only need the single m_my_minibatch_indices list.
  m_minibatch_indices = &(m_reader->get_minibatch_indices());
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

}  // namespace lbann
