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

#ifndef __DATA_STORE_MERGE_SAMPLES_HPP__
#define __DATA_STORE_MERGE_SAMPLES_HPP__

#include "lbann/data_store/generic_data_store.hpp"


namespace lbann {

/**
 * todo
 */

class data_store_pilot2_molecular;

class data_store_merge_samples : public generic_data_store {
 public:

  //! ctor
  data_store_merge_samples(lbann_comm *comm, generic_data_reader *reader, model *m); 

  //! copy ctor
  data_store_merge_samples(const data_store_merge_samples&) = default;

  //! operator=
  data_store_merge_samples& operator=(const data_store_merge_samples&) = default;

  data_store_merge_samples * copy() const override { return new data_store_merge_samples(*this); }

  //! dtor
  ~data_store_merge_samples() override;

  using generic_data_store::get_data_buf;
  void get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx = 0) override {}

  void setup() override;

 protected :

  void exchange_data() override;

  /// this contains a concatenation of the indices in m_minibatch_indices
  /// (see: generic_data_reader.hpp)
  std::vector<int> m_my_minibatch_indices;

  std::vector<data_store_pilot2_molecular*> m_subsidiary_stores;


  /// when running in in-memory mode, this buffer will contain
  /// the concatenated data
  //std::vector<unsigned char> m_data;

  /// allocate mem for m_data
  //void allocate_memory(); 

  //void read_files();

  /// will contain data to be passed to the data_reader
  //std::vector<std::vector<unsigned char> > m_my_data;

  /// maps indices wrt shuffled indices to indices in m_my_data
  //std::unordered_map<size_t, size_t> m_my_data_hash;

  MPI_Win m_win;
};

}  // namespace lbann

#endif  // __DATA_STORE_MERGE_SAMPLES_HPP__
