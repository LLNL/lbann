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

#ifndef __GENERIC_DATA_STORE_HPP__
#define __GENERIC_DATA_STORE_HPP__


#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace lbann {

class lbann_comm;
class model;

/**
 * todo
 */

class data_store_conduit {
 public:

  //! ctor
  data_store_conduit(data_store_conduit *reader, model *m);

  //! copy ctor
  data_store_conduit(const data_store_conduit&) = default;

  //! operator=
  data_store_conduit& operator=(const data_store_conduit&) = default;

  //! dtor
  virtual ~data_store_conduit() {}

  virtual data_store_conduit * copy() const = 0;

  /// called by data_reader
  virtual void setup();

  /// called by generic_data_reader::update()
  virtual void set_shuffled_indices(const std::vector<int> *indices, bool exchange_indices = true);

  const std::string & get_name() const {
    return m_name;
  }

  void set_name(std::string name) {
    m_name = name;
  }

/* dah - not sure if this is needed or how it works;
 * this is for compound_data_reader hierarchy
  void set_is_subsidiary_store() {
    m_is_subsidiary_store = true;
  }

  bool is_subsidiary_store() const {
    return m_is_subsidiary_store;
  }
*/

protected :

  // number of times exchange_data is called
  int m_n;

  void exchange_data() {}

  data_store_conduit *m_reader;

  lbann_comm *m_comm;

  std::string m_name;

  /// this processor's rank
  int  m_rank;

  /// number of procs in the model
  int  m_np;

  bool m_in_memory;

  bool m_master;

  const std::vector<int> *m_shuffled_indices;

  model *m_model;

  /// maps an index to the processor that owns the associated data
  std::unordered_map<int, int> m_owner;

  /// fills in m_owner
  virtual void build_index_owner() {}

  /// as of now, only applicable to merge_features and merge_samples
  /// as of now, only applicable to merge_features and merge_samples
  //bool m_is_subsidiary_store;

  bool m_is_setup;
  bool m_verbose;
};

}  // namespace lbann

#endif  // __GENERIC_DATA_STORE_HPP__
