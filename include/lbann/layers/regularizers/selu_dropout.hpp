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

#ifndef LBANN_LAYER_REGULARIZER_SELU_DROPOUT_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_SELU_DROPOUT_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Scaled dropout for use with SELU activations.
 *
 *  A default keep probability of 0.95 is recommended. See:
 *
 *  Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp
 *  Hochreiter. "Self-normalizing neural networks." In Advances in
 *  Neural Information Processing Systems, pp. 971-980. 2017.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class selu_dropout final : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The tensor type expected in this object. */
  using CPUMatrixType = El::Matrix<TensorDataType, El::Device::CPU>;

  ///@}

public:
  /** Keep units with probabiliy keep_prob. */
  selu_dropout(TensorDataType keep_prob = El::To<TensorDataType>(0.95),
               TensorDataType alpha =
                 El::To<TensorDataType>(1.6732632423543772848170429916717),
               TensorDataType scale =
                 El::To<TensorDataType>(1.0507009873554804934193349852946));

  selu_dropout(const selu_dropout& other);

  selu_dropout& operator=(const selu_dropout& other);

  ~selu_dropout() final;

  selu_dropout* copy() const final;

  std::string get_type() const final;

  data_layout get_data_layout() const final;

  El::Device get_device_allocation() const final;

  bool can_run_inplace() const override { return true; }

  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  void setup_dims(DataReaderMetaData& dr_metadata) final;

  void setup_data(size_t max_mini_batch_size) final;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  /** Drop out units in forward propagation. */
  void fp_compute() final;

  /** Adjust gradients for dropout in backprop. */
  void bp_compute() final;

private:
  /** Alpha prime, the low-variance saturation point. */
  TensorDataType m_alpha_prime;
  /** Affine scaling parameter to keep mean/variance at desired value. */
  TensorDataType m_a;
  /** Affine additive parameter to keep mean/variance at desired value. */
  TensorDataType m_b;
  /** Probability of keeping each unit. */
  TensorDataType m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  AbsDistMatrixType* m_mask;
};

LBANN_DEFINE_LAYER_BUILDER(selu_dropout);

#ifndef LBANN_SELU_DROPOUT_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class selu_dropout<T, data_layout::DATA_PARALLEL, Device>;   \
  extern template class selu_dropout<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SELU_DROPOUT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_SELU_DROPOUT_HPP_INCLUDED
