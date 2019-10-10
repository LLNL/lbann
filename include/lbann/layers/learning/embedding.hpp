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

/** @brief Lookup table to vectors of fixed size.
 *
 *  Takes a scalar input, interprets it as an index, and outputs the
 *  corresponding vector. The number of embedding vectors and the size
 *  of vectors are fixed. If the index is out-of-range, then the
 *  output is a vector of zeros.
 *
 *  The embedding vectors are stored in an
 *  @f$ \text{embedding_dim} \times \text{num_embeddings} @f$
 *  weights matrix. Note that this is the transpose of the weights in
 *  the PyTorch embedding layer.
 */
template <data_layout Layout, El::Device Device>
class embedding_layer : public Layer {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "embedding layer only supports data parallel layout");
  static_assert(Device == El::Device::CPU,
                "embedding layer only supports CPU");
public:

  /**
   *  @param comm           LBANN communicator.
   *  @param num_embeddings Size of dictionary of embeddings.
   *  @param embedding_dim  Size of embedding vectors.
   */
  embedding_layer(lbann_comm* comm,
                  size_t num_embeddings,
                  size_t embedding_dim)
    : Layer(comm),
      m_num_embeddings{num_embeddings},
      m_embedding_dim{embedding_dim} {
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
    desc.add("Num embeddings", m_num_embeddings);
    desc.add("Embedding dim", m_embedding_dim);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;
  /** Gradient w.r.t. embedding weights. */
  StarMat<El::Device::CPU> m_dictionary_gradient;

};

#ifndef LBANN_EMBEDDING_LAYER_INSTANTIATE
extern template class embedding_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
#endif // LBANN_EMBEDDING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
