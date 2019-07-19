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

#ifndef LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

template <data_layout Layout, El::Device Device>
class embedding_layer : public Layer {
public:

  embedding_layer(lbann_comm* comm,
                  El::Int dictionary_size,
                  El::Int embedding_size)
    : Layer(comm),
      m_dictionary_size{dictionary_size},
      m_embedding_size{embedding_size} {
    static_assert(Layout == data_layout::DATA_PARALLEL,
                  "embedding layer only supports data parallel layout");
    static_assert(Device == El::Device::CPU,
                  "embedding layer only supports CPU");
  }

  embedding_layer(const embedding_layer& other) = default;
  embedding_layer& operator=(const embedding_layer& other) = default;
  ~embedding_layer() = default;

  embedding_layer* copy() const override {
    return new embedding_layer(*this);
  }

  std::string get_type() const override { return "embedding"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = Layer::get_description();
    desc.add("Dictionary size", m_dictionary_size);
    desc.add("Embedding size", m_embedding_size);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  El::Int m_dictionary_size;
  El::Int m_embedding_size;
  StarMat<El::Device::CPU> m_dictionary_gradient;

};

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
