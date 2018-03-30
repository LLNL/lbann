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

#ifndef __DATA_STORE_PILOT2_MOLECULAR_HPP__
#define __DATA_STORE_PILOT2_MOLECULAR_HPP__

#include "lbann/data_store/generic_data_store.hpp"
#include <unordered_map>

namespace lbann {

class pilot2_molecular_reader;
class data_store_merge_samples;

/**
 * todo
 */

class data_store_pilot2_molecular : public generic_data_store {
 public:

  //! ctor
  data_store_pilot2_molecular(generic_data_reader *reader, model *m);

  //! copy ctor
  data_store_pilot2_molecular(const data_store_pilot2_molecular&) = default;

  //! operator=
  data_store_pilot2_molecular& operator=(const data_store_pilot2_molecular&) = default;

  data_store_pilot2_molecular * copy() const override { return new data_store_pilot2_molecular(*this); }

  //! dtor
  ~data_store_pilot2_molecular() override;

  void get_data_buf(int data_id, int tid, std::vector<double> *&buf) override; 

  void setup() override;

  /// needed to support data_reader_merge_samples (compound reader)
  void clear_minibatch_indices() {
    m_my_minibatch_indices_v.clear();
  }

  /// needed to support data_reader_merge_samples (compound reader)
  void add_minibatch_index(int idx) {
    m_my_minibatch_indices_v.push_back(idx);
  }

  /// needed to support data_reader_merge_samples (compound reader)
  void set_no_shuffle() {
    m_shuffle = false;
  }

 protected :

   friend data_store_merge_samples;

   pilot2_molecular_reader *m_pilot2_reader;

  /// fills in m_data and m_offsets
  void construct_data_store();
  /// the data store. Note that this will break if word size = 4;
  /// only meaningful on the owning processor
  std::vector<double> m_data;
  /// called by construct_data_store()
  void fill_in_data(
    const int data_id, 
    size_t &jj, 
    const int num_samples_per_frame, 
    const int num_features, 
    double *features);

  /// bcasts m_offsets from owning processor to other procs
  void bcast_offsets();
  /// maps a shuffled index to the offset, wrt m_data, where the
  /// molecule's features (and neighboring molecules) begins
  std::unordered_map<int, int> m_offsets;

  /// maps: a shuffled index to the corresponding molecule's neighbors' indices
  std::unordered_map<int, std::vector<int> > m_neighbors;
  /// fills in m_my_ids
  void build_nabor_map();

  /// fills in m_my_molecules via one-sided MPI calls
  void exchange_data() override;

  /// contains the data of all molecules required by this processor
  /// to execute one epoch
  /// @todo: if num_nabors gets large enough this may need to be re-designed
  std::vector<std::vector<double> > m_my_molecules;
  /// maps moleclue keys in m_my_molecules
  std::unordered_map<size_t, size_t> m_molecule_hash;

  /// the buffers that will be passed to data_readers::fetch_datum
  std::vector<std::vector<double> > m_data_buffer;

  /// the process that "owns" the data, i.e, this is the only process
  /// whose m_reader will load data from disk
  int m_owner_rank;

  /// true if this processor "owns" the data
  bool m_owner;

  /// support for data_store_merge_samples
  bool m_shuffle;

  MPI_Win m_win;
};

}  // namespace lbann

#endif  // __DATA_STORE_PILOT2_MOLECULAR_HPP__
