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

#include "lbann/data_store/data_store_merge_samples.hpp"
#include "lbann/data_store/data_store_pilot2_molecular.hpp"
#include "lbann/data_readers/data_reader_pilot2_molecular.hpp"
#include "lbann/data_readers/data_reader_merge_samples.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

data_store_merge_samples::data_store_merge_samples(lbann_comm *comm, generic_data_reader *reader, model *m) :
    generic_data_store(reader, m) {
  set_name("data_store_merge_samples");
}


data_store_merge_samples::~data_store_merge_samples() {
  MPI_Win_free( &m_win );
}

void data_store_merge_samples::setup() {
  if (m_rank == 0) std::cerr << "STARTING data_store_merge_samples::setup()\n"; 
  //double tm1 = get_time();

  generic_data_store::setup();

/*
  bool run_tests = false;
  if (options::get()->has_bool("test_data_store") && options::get()->get_bool("test_data_store")) {
    run_tests = true;
  }
  */

  if (m_rank == 0) {
    std::cout << "starting data_store_merge_samples::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }
  
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    //sanity check
    data_reader_merge_samples *reader = dynamic_cast<data_reader_merge_samples*>(m_reader);
    if (reader == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<merge_samples_reader*>(m_reader) failed";
      throw lbann_exception(err.str());
    }


    // get list of indices used in calls to generic_data_reader::fetch_data
    get_minibatch_index_vector();

    std::vector<generic_data_reader*> &readers = reader->get_data_readers();
    for (size_t j=0; j<readers.size(); j++) {
      readers[j]->get_data_store();
      pilot2_molecular_reader *pilot2_reader = dynamic_cast<pilot2_molecular_reader*>(m_reader);
      generic_data_store *store = pilot2_reader->get_data_store();
      data_store_pilot2_molecular *s = dynamic_cast<data_store_pilot2_molecular*>(store);
      s->clear_minibatch_indices();
      m_subsidiary_stores.push_back(s);
    }

    for (auto t : m_subsidiary_stores) {
      t->set_no_shuffle();
    }

    const std::vector<int> &num_samples_psum = reader->get_num_samples_psum();
    for (auto data_id : m_my_minibatch_indices_v) {
      for (size_t i = 0; i < m_subsidiary_stores.size(); ++i) {
        if (data_id < num_samples_psum[i + 1]) {
          data_id -= num_samples_psum[i];
          m_subsidiary_stores[i]->add_minibatch_index(data_id);
        }
      }
    }
  }
}

void data_store_merge_samples::exchange_data() {
  //for (auto t : m_subsidiary_stores) {
    
}

}  // namespace lbann
