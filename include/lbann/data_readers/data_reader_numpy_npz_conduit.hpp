////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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


#ifndef LBANN_DATA_READER_NUMPY_NPZ_CONDUIT_HPP
#define LBANN_DATA_READER_NUMPY_NPZ_CONDUIT_HPP

#include "lbann/data_readers/data_reader.hpp"
#include <cnpy.h>

namespace lbann {
  /**
   * Data reader for data stored in numpy (.npz) files that are encapsulated .
   * in conduit::Nodes
   */
  class numpy_npz_conduit_reader : public generic_data_reader {
 public:
  numpy_npz_conduit_reader(const bool shuffle);
  // These need to be explicit because of some issue with the cnpy copy
  // constructor/assignment operator not linking correctly otherwise.
  // dah -- ??
  numpy_npz_conduit_reader(const numpy_npz_conduit_reader&);
  numpy_npz_conduit_reader& operator=(const numpy_npz_conduit_reader&);
  ~numpy_npz_conduit_reader() override {}

  numpy_npz_conduit_reader* copy() const override { return new numpy_npz_conduit_reader(*this); }

  void copy_members(const numpy_npz_conduit_reader& rhs);

  std::string get_type() const override {
    return "numpy_npz_conduit_reader";
  }

  /// Set whether to fetch labels.
  void set_has_labels(bool b) { m_has_labels = b; }
  /// Set whether to fetch responses.
  void set_has_responses(bool b) { m_has_responses = b; }
  /// Set a scaling factor for int16 data.
  void set_scaling_factor_int16(DataType s) { m_scaling_factor_int16 = s; }

  void load() override;

  void set_num_labels(int n) { m_num_labels = n; }
  int get_num_labels() const override { return m_num_labels; }
  int get_num_responses() const override { return get_linearized_response_size(); }
  int get_linearized_data_size() const override { return m_num_features; }
  int get_linearized_label_size() const override { return m_num_labels; }
  int get_linearized_response_size() const override { return m_num_response_features; }
  const std::vector<int> get_data_dims() const override { return m_data_dims; }

  protected:
    void preload_data_store() override;

    bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
    bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
    bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

    /// Number of samples.
    int m_num_samples = 0;
    /// Number of features in each sample.
    int m_num_features = 0;
    /// Number of label classes.
    int m_num_labels = 0;
    /// Number of features in each response.
    int m_num_response_features = 0;
    /// Whether to fetch a label from the last column.
    bool m_has_labels = true;
    /// Whether to fetch a response from the last column.
    bool m_has_responses = true;

    std::vector<int> m_data_dims;
    int m_data_word_size = 0;
    size_t m_response_word_size = 0;

    // A constant to be multiplied when data is converted
    // from int16 to DataType.
    DataType m_scaling_factor_int16 = 1.0;

    // fills in: m_num_samples, m_num_features, m_num_response_features,
    // m_data_dims, m_data_word_size, m_response_word_size
    void fill_in_metadata();

    std::vector<std::string> m_filenames;
  };

}  // namespace lbann

#endif  // LBANN_DATA_READER_NUMPY_NPZ_CONDUIT_HPP
