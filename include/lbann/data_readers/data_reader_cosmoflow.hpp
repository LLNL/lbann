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
// lbann_data_reader_cosmoflow .hpp .cpp - data_reader class for CosmoFlow
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_COSMOFLOW_HPP
#define LBANN_DATA_READER_COSMOFLOW_HPP

#include "data_reader.hpp"
#include "data_reader_numpy.hpp"
#include <cnpy.h>

namespace lbann {
/**
 * A data reader class for datasets which composed of .npz files,
 * expecially for CosmoFlow datasets.
 */
class cosmoflow_reader : public generic_data_reader {
 public:
  cosmoflow_reader(const bool shuffle);
  // These need to be explicit because of some issue with the cnpy copy
  // constructor/assignment operator not linking correctly otherwise.
  cosmoflow_reader(const cosmoflow_reader&);
  cosmoflow_reader& operator=(const cosmoflow_reader&);
  ~cosmoflow_reader() override {}

  cosmoflow_reader* copy() const override { return new cosmoflow_reader(*this); }

  std::string get_type() const override {
    return "cosmoflow_reader";
  }

  /// Set a scaling factor for int16 data.
  void set_scaling_factor_int16(DataType s) { m_scaling_factor_int16 = s; }

  /// Set paths to .npz files.
  void set_npz_paths(const std::vector<std::string> npz_paths) { m_npz_paths = npz_paths; }

  void load() override;

  int get_num_responses() const override { return get_linearized_response_size(); }
  int get_linearized_data_size() const override { return m_num_features; }
  int get_linearized_response_size() const override { return m_num_response_features; }
  const std::vector<int> get_data_dims() const override { return m_data_dims; }

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

  std::pair<cnpy::NpyArray, int> prepare_npz_file(const int data_id,
                                                  const std::string key);

  /// Number of samples.
  int m_num_samples_total = 0;
  /// Number of features in each sample.
  int m_num_features = 0;
  /// Number of features in each response.
  int m_num_response_features = 0;
  /// Number of samples in each .npz file and their prefix sum.
  std::vector<int> m_num_samples;
  std::vector<int> m_num_samples_prefix;
  /// Shape of each sample. This data reader assumes that all of the
  /// samples have the same shape.
  std::vector<int> m_data_dims;

  // Paths to .npz files.
  std::vector<std::string> m_npz_paths;

  // A constant to be multiplied when data is converted
  // from int16 to DataType.
  DataType m_scaling_factor_int16 = 1.0;

 private:
  // Keys to retrieve data and responses from a given .npz file.
  static const std::string NPZ_KEY_DATA, NPZ_KEY_RESPONSES;

};

}  // namespace lbann

#endif  // LBANN_DATA_READER_COSMOFLOW_HPP
