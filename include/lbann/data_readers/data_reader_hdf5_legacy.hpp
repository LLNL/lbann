////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////
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
#ifndef LBANN_DATA_READER_HDF5_HPP
#define LBANN_DATA_READER_HDF5_HPP
#include "conduit/conduit.hpp"
#include "data_reader_image.hpp"
#include "hdf5.h"

namespace lbann {
/**
 * Data reader for data stored in HDF5 files. This data reader was
 * designed to work with Distconv. This currently has two different
 * modes:
 * * Datasets with 3D data and a few numbers of responses:
 *   This mode assumes a 3D cube dataset such as the CosmoFlow dataset.
 *   This requires set_has_responses to be called on setup.
 * * Datasets with 3D data and 3D labels:
 *   This mode assumes 3D cubes with corresponding 3D label tensors
 *   such as the LiTS dataset. This requires set_has_labels to be
 *   called on setup, and label_reconstruction should be used for the
 *   input layer.
 *
 * Each HDF5 file should contain hdf5_key_data, hdf5_key_labels, and
 * hdf5_key_responses keys to read data, labels and responses
 * respectively.
 */
template <typename TensorDataType>
class hdf5_reader : public generic_data_reader
{
public:
  hdf5_reader(const bool shuffle,
              const std::string key_data,
              const std::string key_label,
              const std::string key_responses,
              const bool hyperslab_labels);
  hdf5_reader(const hdf5_reader&);
  hdf5_reader& operator=(const hdf5_reader&);
  ~hdf5_reader() override {}

  hdf5_reader* copy() const override { return new hdf5_reader(*this); }

  void copy_members(const hdf5_reader& rhs);

  std::string get_type() const override { return "data_reader_hdf5_images"; }

  bool supports_background_io() override { return false; }

  // void set_input_params(int width, int height, int depth, int num_ch, int
  // num_labels);
  void load() override;
  void set_hdf5_paths(const std::vector<std::string> hdf5_paths)
  {
    m_file_paths = hdf5_paths;
  }

  void set_num_responses(const size_t num_responses)
  {
    m_all_responses.resize(num_responses);
  }

  int get_num_labels() const override
  {
    if (!this->has_labels()) {
      return generic_data_reader::get_num_labels();
    }
    // This data reader currently assumes that the shape of the label
    // tensor is the same to the data tensor.
    return m_num_features;
  }
  int get_num_responses() const override
  {
    if (!this->has_responses()) {
      return generic_data_reader::get_num_responses();
    }
    return get_linearized_response_size();
  }
  int get_linearized_data_size() const override { return m_num_features; }
  int get_linearized_label_size() const override
  {
    if (!this->has_labels()) {
      return generic_data_reader::get_linearized_label_size();
    }
    // This data reader currently assumes that the shape of the label
    // tensor is the same to the data tensor.
    return m_num_features;
  }
  int get_linearized_response_size() const override
  {
    if (!this->has_responses()) {
      return generic_data_reader::get_linearized_response_size();
    }
    return m_all_responses.size();
  }
  const std::vector<El::Int> get_data_dims() const override
  {
    return m_data_dims;
  }

#ifdef LBANN_HAS_DISTCONV
  bool is_tensor_shuffle_required() const override { return false; }
#endif // LBANN_HAS_DISTCONV

protected:
  void read_hdf5_hyperslab(hsize_t h_data,
                           hsize_t filespace,
                           int rank,
                           TensorDataType* sample);
  void
  read_hdf5_sample(int data_id, TensorDataType* sample, TensorDataType* labels);
  // void set_defaults() override;
  void load_sample(conduit::Node& node, int data_id);
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  void fetch_datum_conduit(Mat& X, int data_id);
  bool fetch_data_field(data_field_type data_field,
                        CPUMat& Y,
                        int data_id,
                        int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;
  hid_t get_hdf5_data_type() const;
  conduit::DataType get_conduit_data_type(conduit::index_t num_elements) const;

  int m_image_depth = 0;
  size_t m_num_features;
  std::vector<float> m_all_responses;
  std::vector<std::string> m_file_paths;
  MPI_Comm m_comm;
  std::vector<El::Int> m_data_dims;
  std::vector<hsize_t> m_hyperslab_dims;
  hid_t m_fapl;
  hid_t m_dxpl;
  MPI_Comm m_response_gather_comm;
  bool m_use_data_store;
  std::string m_key_data, m_key_labels, m_key_responses;
  bool m_hyperslab_labels;

private:
  static const std::string HDF5_KEY_DATA, HDF5_KEY_LABELS, HDF5_KEY_RESPONSES;
};
} // namespace lbann
#endif // LBANN_DATA_READER_HDF5_HPP
