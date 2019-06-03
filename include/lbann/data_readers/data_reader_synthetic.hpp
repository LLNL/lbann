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

#ifndef LBANN_DATA_READER_SYNTHETIC_HPP
#define LBANN_DATA_READER_SYNTHETIC_HPP

#include "data_reader.hpp"

namespace lbann {

/**
 * Data reader for generating random samples.
 * Samples are different every time.
 */
class data_reader_synthetic : public generic_data_reader {
 public:
  //@todo: add what data distribution to use
  data_reader_synthetic(int num_samples, int num_features, bool shuffle = true);
  data_reader_synthetic(int num_samples, std::vector<int> dims,
                        int num_labels, bool shuffle = true);
  data_reader_synthetic(int num_samples, std::vector<int> dims,
                        std::vector<int> response_dims, bool shuffle = true);
  data_reader_synthetic(const data_reader_synthetic&) = default;
  data_reader_synthetic& operator=(const data_reader_synthetic&) = default;
  ~data_reader_synthetic() override {}
  data_reader_synthetic* copy() const override {
    return new data_reader_synthetic(*this);
  }
  std::string get_type() const override {
    return "data_reader_synthetic";
  }

  void load() override;

  int get_linearized_data_size() const override {
    return std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1,
                           std::multiplies<int>());
  }
  int get_linearized_label_size() const override {
    return m_num_labels;
  }
  int get_linearized_response_size() const override {
    return std::accumulate(m_response_dimensions.begin(),
                           m_response_dimensions.end(), 1,
                           std::multiplies<int>());
  }

  const std::vector<int> get_data_dims() const override {
    return m_dimensions;
  }

  int get_num_labels() const override { return m_num_labels; }
  int get_num_responses() const override {
    return get_linearized_response_size();
  }

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

 private:
  /** Number of samples in the dataset. */
  int m_num_samples;
  /** Number of labels in the dataset. */
  int m_num_labels;
  /** Shape of the data. */
  std::vector<int> m_dimensions;
  /** Shape of the responses. */
  std::vector<int> m_response_dimensions;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_SYNTHETIC_HPP
