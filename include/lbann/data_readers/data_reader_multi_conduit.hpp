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

#ifndef LBANN_DATA_READER_MULTI_CONDUIT_HPP
#define LBANN_DATA_READER_MULTI_CONDUIT_HPP

#include "compound_data_reader.hpp"

namespace lbann {

/**
 * Data reader for merging the samples from multiple data readers into a
 * single dataset.
 */
class data_reader_multi_conduit : public generic_compound_data_reader {
 public:
  data_reader_multi_conduit(std::vector<generic_data_reader*> data_readers,
                            bool shuffle = true);
  data_reader_multi_conduit(const data_reader_multi_conduit&);

  data_reader_multi_conduit& operator=(const data_reader_multi_conduit&);

  ~data_reader_multi_conduit() override;

  data_reader_multi_conduit* copy() const override {
    return new data_reader_multi_conduit(*this);
  }

  std::string get_type() const override {
    return "data_reader_multi_conduit";
  }

  /// Load subsidiary data readers.
  void load() override;

  int get_num_labels() const override { return m_data_readers[0]->get_num_labels(); }

  int get_linearized_data_size() const override {
    return m_data_readers[0]->get_linearized_data_size();
  }
  int get_linearized_label_size() const override {
    return m_data_readers[0]->get_linearized_label_size();
  }
  const std::vector<int> get_data_dims() const override {
    return m_data_readers[0]->get_data_dims();
  }

  /// support for data store functionality
  /* todo
  const std::vector<int> & get_num_samples_psum() {
    return m_num_samples_psum;
  }
  */

  /// sets up a data_store.
  void setup_data_store(model *m) override;

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) override;

  /// Partial sums of the number of samples in each reader.
  //std::vector<int> m_num_samples_psum;

  /// support for data store functionality; load() will call
  /// this method when using data store
  // todo
  //void load_using_data_store();

  /// code common to both load() and load_using_data_store()
//  void setup_indices(int num_samples);

  /// code common to both load() and load_using_data_store()
//  size_t compute_num_samples_psum();
  
  /// code common to both load() and load_using_data_store()
//  void sanity_check_for_consistency(int num_labels, int data_size, int label_size, const std::vector<int> &data_dims);

};

}  // namespace lbann

#endif  // LBANN_DATA_READER_NULTI_CONDUIT_HPP
