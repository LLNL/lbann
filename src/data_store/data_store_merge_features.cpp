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

#include "lbann/data_store/data_store_merge_features.hpp"
#include "lbann/data_store/data_store_csv.hpp"
#include "lbann/data_readers/data_reader_merge_features.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

data_store_merge_features::data_store_merge_features(generic_data_reader *reader, model *m) :
    generic_data_store(reader, m) {
  set_name("data_store_merge_features");
}


data_store_merge_features::~data_store_merge_features() {
}

void data_store_merge_features::exchange_data() {
  for (auto s : m_subsidiary_stores) {
    data_store_csv *store = dynamic_cast<data_store_csv*>(s);
    store->set_shuffled_indices(m_shuffled_indices, false);
    store->exchange_data();
  }
}

void data_store_merge_features::setup() {
  double tm1 = get_time();
  if (m_master) {
    std::cerr << "starting data_store_merge_features::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }

  generic_data_store::setup();

  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
  } 
  
  else {
    //sanity check
    data_reader_merge_features *reader = dynamic_cast<data_reader_merge_features*>(m_reader);
    if (reader == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<merge_features_reader*>(m_reader) failed";
      throw lbann_exception(err.str());
    }

    // get list of indices used in calls to generic_data_reader::fetch_data
    if (m_master) std::cerr << "calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();

    if (m_master) std::cerr << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "calling exchange_mb_indices()\n";
    exchange_mb_indices();

    std::vector<generic_data_reader*> &readers = reader->get_data_readers();
    m_subsidiary_stores.reserve(readers.size());
    for (auto r : readers) {
      data_store_csv *store = new data_store_csv(r, m_model);
      m_subsidiary_stores.push_back(store);
      r->set_data_store(store);
      store->set_is_subsidiary_store();
      store->set_minibatch_indices(get_minibatch_indices());
      store->set_all_minibatch_indices(get_all_minibatch_indices());
      store->set_minibatch_indices_v(get_minibatch_indices_v());
      store->set_datastore_indices(get_datastore_indices());
      store->setup();
      store->set_shuffled_indices(m_shuffled_indices, false);
      store->populate_datastore();
      store->exchange_data();
    }
  }
  if (m_master) {
    std::cerr << "TIME for data_store_merge_features setup: " << get_time() - tm1 << "\n";
  }
}

}  // namespace lbann
