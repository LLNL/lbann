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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SORT_HPP_INCLUDED
#define LBANN_LAYER_SORT_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Sort entries in each mini-batch sample. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class sort_layer : public transform_layer {
 public:

  sort_layer(lbann_comm *comm) : transform_layer(comm) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "sort layer only supports DATA_PARALLEL");
  }
  sort_layer(const sort_layer& other)
    : transform_layer(other),
      m_indices(other.m_indices ? other.m_indices->Copy() : nullptr) {}
  sort_layer& operator=(const sort_layer& other) {
    transform_layer::operator=(other);
    m_indices.reset(other.m_indices ? other.m_indices->Copy() : nullptr);
    return *this;
  }
  
  sort_layer* copy() const override { return new sort_layer(*this); }
  std::string get_type() const override { return "sort"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_dims() override {
    set_output_dims(get_input_dims());
    transform_layer::setup_dims();
  }
  
  void setup_matrices(const El::Grid& grid) override {
    transform_layer::setup_matrices(grid);
    const auto& dist = get_activations().DistData();
    m_indices.reset(El::AbstractDistMatrix<El::Int>
                    ::Instantiate(*dist.grid,
                                  dist.root,
                                  dist.colDist,
                                  dist.rowDist,
                                  (dist.blockHeight == 1
                                   && dist.blockWidth == 1 ?
                                   El::ELEMENT : El::BLOCK),               
                                  dist.device));
#ifdef LBANN_HAS_GPU
    // Allocate GPU memory with the CUDA API
    if (m_indices->GetLocalDevice() == El::Device::GPU) {
      m_indices->Matrix().SetMemoryMode(0);
    }
#endif //LBANN_HAS_CUDA
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    transform_layer::setup_data();
    const auto& output = get_activations();
    m_indices->Empty(false);
    m_indices->AlignWith(output);
    m_indices->Resize(output.Height(), output.Width());
  }
  
  void fp_compute() override;
  void bp_compute() override;

 private:

  std::unique_ptr<El::AbstractDistMatrix<El::Int>> m_indices;
  
};

} // namespace lbann

#endif // LBANN_LAYER_SORT_HPP_INCLUDED
