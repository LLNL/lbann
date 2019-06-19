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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SORT_HPP_INCLUDED
#define LBANN_LAYER_SORT_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** @brief Sort tensor entries. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class sort_layer : public transform_layer {
 public:

  sort_layer(lbann_comm *comm, bool descending = false)
    : transform_layer(comm), m_descending(descending) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "sort layer only supports DATA_PARALLEL");
  }
  sort_layer(const sort_layer& other)
    : transform_layer(other),
      m_descending(other.m_descending) {
    if (other.m_indices) {
      switch (other.m_indices->GetDevice()) {
      case El::Device::CPU:
        m_indices.reset(new El::Matrix<El::Int, El::Device::CPU>());
        break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU:
        m_indices.reset(new El::Matrix<El::Int, El::Device::GPU>());
        break;
#endif // LBANN_HAS_GPU
      default: LBANN_ERROR("invalid device");
      }
      El::Copy(*other.m_indices, *m_indices);
    }
  }
  sort_layer& operator=(const sort_layer& other) {
    transform_layer::operator=(other);
    m_descending = other.m_descending;
    if (!other.m_indices) {
      m_indices.reset(nullptr);
    } else {
      switch (other.m_indices->GetDevice()) {
      case El::Device::CPU:
        m_indices.reset(new El::Matrix<El::Int, El::Device::CPU>());
        break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU:
        m_indices.reset(new El::Matrix<El::Int, El::Device::GPU>());
        break;
#endif // LBANN_HAS_GPU
      default: LBANN_ERROR("invalid device");
      }
      El::Copy(*other.m_indices, *m_indices);
    }
    return *this;
  }

  sort_layer* copy() const override { return new sort_layer(*this); }
  std::string get_type() const override { return "sort"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto&& desc = transform_layer::get_description();
    desc.add("Descending", m_descending);
    return desc;
  }

 protected:

  void setup_dims() override {
    transform_layer::setup_dims();
    set_output_dims(get_input_dims());
  }

  void setup_matrices(const El::Grid& grid) override {
    transform_layer::setup_matrices(grid);
    const auto& dist = get_activations().DistData();
    switch (dist.device) {
    case El::Device::CPU:
      m_indices.reset(new El::Matrix<El::Int, El::Device::CPU>());
      break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_indices.reset(new El::Matrix<El::Int, El::Device::GPU>());
      m_indices->SetMemoryMode(0); // Allocate GPU memory with the CUDA API
      break;
#endif // LBANN_HAS_GPU
    default: LBANN_ERROR("invalid device");
    }
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    transform_layer::fp_setup_outputs(mini_batch_size);
    const auto& output = get_activations();
    m_indices->Resize(output.LocalHeight(), output.LocalWidth());
  }

  void fp_compute() override;
  void bp_compute() override;

 private:

  /** Whether values are sorted by descending order. */
  bool m_descending;

  /** Input indices corresponding to output entries.
   *  @todo Switch to distributed integer matrix once it's supported
   *  in Hydrogen.
   */
  std::unique_ptr<El::AbstractMatrix<El::Int>> m_indices;

};

} // namespace lbann

#endif // LBANN_LAYER_SORT_HPP_INCLUDED
