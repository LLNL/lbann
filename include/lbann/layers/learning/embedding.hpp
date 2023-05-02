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

#ifndef LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/memory.hpp"

namespace lbann {

/** @brief Lookup table to vectors of fixed size.
 *
 *  Each input value is interpreted as an index and the corresponding
 *  embedding vector is output. Thus, given an input vector of length
 *  @f$ \text{sequence\_length} @f$, the output is a
 *  @f$ \text{sequence\_length} \times \text{embedding\_dim} @f$ tensor.
 *  If an index is out-of-range, then corresponding output is a vector
 *  of zeros.
 *
 *  The embedding vectors are stored in an
 *  @f$ \text{embedding\_dim} \times \text{num\_embeddings} @f$
 *  weights matrix. Note that this is the transpose of the weights in
 *  the PyTorch embedding layer.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class embedding_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "embedding layer only supports data parallel layout");

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The concrete optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  ///@}

public:
  /** @brief Constructor
   *  @param num_embeddings Size of dictionary of embeddings.
   *  @param embedding_dim  Size of embedding vectors.
   *  @param padding_idx    If set, then the corresponding embedding
   *                        vector is initialized with zeros. The
   *                        objective function gradient w.r.t. this
   *                        embedding vector is always zero.
   */
  embedding_layer(size_t num_embeddings,
                  size_t embedding_dim,
                  El::Int padding_idx = -1);

  embedding_layer(const embedding_layer& other);
  embedding_layer& operator=(const embedding_layer& other);
  ~embedding_layer() = default;

  embedding_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS | PREV_ACTIVATIONS;
  }

  description get_description() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  embedding_layer();

  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;
  /** If the padding index is set, then the corresponding embedding
   *  vector is initialized with zeros. The objective function
   *  gradient w.r.t. this embedding vector is always zero.
   */
  El::Int m_padding_idx;

  /** Gradient w.r.t. embedding weights. */
  std::unique_ptr<AbsDistMatrixType> m_embeddings_grad;
};

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void embedding_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_embedding();
  msg->set_num_embeddings(m_num_embeddings);
  msg->set_embedding_dim(m_embedding_dim);
  msg->mutable_padding_idx()->set_value(m_padding_idx);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
embedding_layer<TensorDataType, Layout, Device>::embedding_layer(
  size_t num_embeddings,
  size_t embedding_dim,
  El::Int padding_idx)
  : data_type_layer<TensorDataType>(nullptr),
    m_num_embeddings{num_embeddings},
    m_embedding_dim{embedding_dim},
    m_padding_idx{padding_idx}
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
embedding_layer<TensorDataType, Layout, Device>::embedding_layer()
  : embedding_layer(0, 0, 0)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
embedding_layer<TensorDataType, Layout, Device>::embedding_layer(
  const embedding_layer<TensorDataType, Layout, Device>& other)
  : data_type_layer<TensorDataType>(other),
    m_num_embeddings{other.m_num_embeddings},
    m_embedding_dim{other.m_embedding_dim},
    m_padding_idx{other.m_padding_idx},
    m_embeddings_grad(other.m_embeddings_grad ? other.m_embeddings_grad->Copy()
                                              : nullptr)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
embedding_layer<TensorDataType, Layout, Device>&
embedding_layer<TensorDataType, Layout, Device>::operator=(
  const embedding_layer<TensorDataType, Layout, Device>& other)
{
  data_type_layer<TensorDataType>::operator=(other);
  m_num_embeddings = other.m_num_embeddings;
  m_embedding_dim = other.m_embedding_dim;
  m_padding_idx = other.m_padding_idx;
  m_embeddings_grad.reset(
    other.m_embeddings_grad ? other.m_embeddings_grad->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
embedding_layer<TensorDataType, Layout, Device>*
embedding_layer<TensorDataType, Layout, Device>::copy() const
{
  return new embedding_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string embedding_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "embedding";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
embedding_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
embedding_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
embedding_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Num embeddings", m_num_embeddings);
  desc.add("Embedding dim", m_embedding_dim);
  desc.add("Padding index", m_padding_idx);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void embedding_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  auto dims = this->get_input_dims();
  dims.push_back(static_cast<int>(m_embedding_dim));
  this->set_output_dims(dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void embedding_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Construct default weights if needed
  // Note: Randomly drawn from normal distribution with mean 0 and
  // standard deviation 1.
  if (!this->has_weights()) {
    auto w = std::make_shared<WeightsType>(*this->get_comm());
    auto init = std::make_unique<normal_initializer<TensorDataType>>(
      El::TypeTraits<TensorDataType>::Zero(),
      El::TypeTraits<TensorDataType>::One());
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->add_weights(w);
    this->m_model->add_weights(std::move(w));
  }
  if (this->num_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",
                this->num_weights(),
                ")");
  }

  // Initialize dictionary
  auto& embeddings = this->get_weights(0);
  auto matrix_dist = this->get_prev_activations().DistData();
  matrix_dist.colDist = El::STAR;
  matrix_dist.rowDist = El::STAR;
  embeddings.set_dims({m_embedding_dim}, {m_num_embeddings});
  embeddings.set_matrix_distribution(matrix_dist);
  embeddings.setup();

  // Zero out embedding vector for padding index
  if (0 <= m_padding_idx &&
      m_padding_idx < static_cast<El::Int>(m_embedding_dim)) {
    // FIXME (trb 06/01/2020): Assuming embedding values have data
    // type that matches this layer. In future, we should abstract
    // this or dynamically dispatch it.
    auto& embedding_values =
      dynamic_cast<AbsDistMatrixType&>(embeddings.get_values());
    std::unique_ptr<AbsDistMatrixType> pad_embedding(
      embedding_values.Construct(embedding_values.Grid(),
                                 embedding_values.Root()));
    El::View(*pad_embedding, embedding_values, El::ALL, El::IR(m_padding_idx));
    El::Zero(*pad_embedding);
  }

  // Initialize gradient w.r.t. embeddings
  {
    auto& embedding_values =
      dynamic_cast<AbsDistMatrixType&>(embeddings.get_values());
    this->m_embeddings_grad.reset(
      embedding_values.Construct(embedding_values.Grid(),
                                 embedding_values.Root()));
    m_embeddings_grad->Resize(m_embedding_dim, m_num_embeddings);
  }
}

LBANN_DEFINE_LAYER_BUILDER(embedding);

#ifndef LBANN_EMBEDDING_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class embedding_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_EMBEDDING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
