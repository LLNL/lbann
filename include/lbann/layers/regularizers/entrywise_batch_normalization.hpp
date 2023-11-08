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

#ifndef LBANN_LAYERS_REGULARIZERS_ENTRYWISE_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_ENTRYWISE_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/memory.hpp"

namespace lbann {

/** @brief Entry-wise batch normalization, including scale/bias
 *
 *  Each input entry is normalized across the mini-batch to have zero
 *  mean and unit standard deviation. This uses the standard approach
 *  of maintaining the running mean and standard deviation (with
 *  exponential decay) for use at test time. See:
 *
 *  Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *  Accelerating Deep Network Training by Reducing Internal Covariate
 *  Shift." In International Conference on Machine Learning,
 *  pp. 448-456. 2015.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class entrywise_batch_normalization_layer
  : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  entrywise_batch_normalization_layer(
    TensorDataType decay = El::To<TensorDataType>(0.9),
    TensorDataType epsilon = El::To<TensorDataType>(1e-5))
    : data_type_layer<TensorDataType>(nullptr),
      m_decay(decay),
      m_epsilon(epsilon)
  {}

  entrywise_batch_normalization_layer(
    const entrywise_batch_normalization_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_decay(other.m_decay),
      m_epsilon(other.m_epsilon),
      m_batch_statistics(
        other.m_batch_statistics ? other.m_batch_statistics->Copy() : nullptr),
      m_batch_statistics_gradient(other.m_batch_statistics_gradient
                                    ? other.m_batch_statistics_gradient->Copy()
                                    : nullptr)
  {}

  entrywise_batch_normalization_layer&
  operator=(const entrywise_batch_normalization_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_batch_statistics.reset(
      other.m_batch_statistics ? other.m_batch_statistics->Copy() : nullptr);
    m_batch_statistics_gradient.reset(
      other.m_batch_statistics_gradient
        ? other.m_batch_statistics_gradient->Copy()
        : nullptr);
    return *this;
  }

  entrywise_batch_normalization_layer* copy() const override
  {
    return new entrywise_batch_normalization_layer(*this);
  }
  std::string get_type() const override
  {
    return "entry-wise batch normalization";
  }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS | WEIGHTS;
  }

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("Decay", m_decay);
    desc.add("Epsilon", m_epsilon);
    return desc;
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_data(size_t max_mini_batch_size) override
  {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Initialize output dimensions
    this->set_output_dims(this->get_input_dims());
    const auto output_dims_ = this->get_output_dims();
    std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());

    // Initialize default weights if none are provided
    if (this->num_weights() > 2) {
      LBANN_ERROR("attempted to setup layer \"",
                  this->get_name(),
                  "\" ",
                  "with an invalid number of weights ",
                  "(found ",
                  this->num_weights(),
                  ", expected 2)");
    }
    this->set_num_weights(2);
    if (!this->has_weights(0)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto init = std::make_unique<constant_initializer<TensorDataType>>(
        El::TypeTraits<TensorDataType>::Zero());
      w->set_name(this->get_name() + "_running_mean");
      w->set_initializer(std::move(init));
      this->set_weights(0, w);
      this->m_model->add_weights(std::move(w));
    }
    if (!this->has_weights(1)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto init = std::make_unique<constant_initializer<TensorDataType>>(
        El::TypeTraits<TensorDataType>::One());
      w->set_name(this->get_name() + "_running_variance");
      w->set_initializer(std::move(init));
      this->set_weights(1, w);
      this->m_model->add_weights(std::move(w));
    }

    // Setup weights
    auto dist = this->get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    auto const num_weights = this->num_weights();
    for (size_t ii = 0; ii < num_weights; ++ii) {
      auto& w = this->get_weights(ii);
      w.set_dims(output_dims);
      w.set_matrix_distribution(dist);
    }

    // Initialize matrices
    m_batch_statistics.reset(AbsDistMatrixType::Instantiate(dist));
    m_batch_statistics_gradient.reset(AbsDistMatrixType::Instantiate(dist));
  }

  void fp_compute() override;
  void bp_compute() override;

private:
  /** Decay rate for the running statistics. */
  TensorDataType m_decay;
  /** Small number to avoid division by zero. */
  TensorDataType m_epsilon;

  /** @brief Current mini-batch statistics.
   *
   *  These are fused for performance when doing non-local batchnorm.
   */
  std::unique_ptr<AbsDistMatrixType> m_batch_statistics;
  /** @brief Gradients w.r.t. current mini-batch statistics.
   *
   * These are fused for performance when doing non-local batchnorm.
   */
  std::unique_ptr<AbsDistMatrixType> m_batch_statistics_gradient;
};

template <typename T, data_layout L, El::Device D>
void entrywise_batch_normalization_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_entrywise_batch_normalization();
  msg->set_decay(m_decay);
  msg->set_epsilon(m_epsilon);
}

LBANN_DEFINE_LAYER_BUILDER(entrywise_batch_normalization);

#ifndef LBANN_ENTRYWISE_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class entrywise_batch_normalization_layer<                   \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    Device>;                                                                   \
  extern template class entrywise_batch_normalization_layer<                   \
    T,                                                                         \
    data_layout::MODEL_PARALLEL,                                               \
    Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_ENTRYWISE_BATCH_NORMALIZATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_ENTRYWISE_BATCH_NORMALIZATION_HPP_INCLUDED
