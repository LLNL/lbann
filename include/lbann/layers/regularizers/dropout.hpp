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

#ifndef LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/dropout.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB
#include "lbann/utils/random_number_generators.hpp"

namespace lbann {

/** @brief Probabilistically drop layer outputs
 *
 *  The weights are multiplied by 1/(keep probability) at training
 *  time. Keep probabilities of 0.5 for fully-connected layers and 0.8
 *  for input layers are good starting points. See:
 *
 *  Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya
 *  Sutskever, and Ruslan Salakhutdinov. "Dropout: a simple way to
 *  prevent neural networks from overfitting." The Journal of Machine
 *  Learning Research 15, no. 1 (2014): 1929-1958.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class dropout : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  /** Keep units with probabiliy keep_prob. */
  dropout(EvalType keep_prob = EvalType(0.5))
    : data_type_layer<TensorDataType>(nullptr),
      m_keep_prob(keep_prob)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {}

  dropout(const dropout& other)
    : data_type_layer<TensorDataType>(other),
      m_keep_prob(other.m_keep_prob),
      m_mask(other.m_mask ? other.m_mask->Copy() : nullptr)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(other.m_tensors_dnn_desc)
#endif // LBANN_HAS_DNN_LIB
  {
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc.set_layer(this);
    m_states = other.m_states;
    m_reserve_space = other.m_reserve_space;
    if (other.m_dropout_dnn_desc != nullptr) {
      setup_dropout_dnn_desc();
    }
#endif // LBANN_HAS_DNN_LIB
  }

  dropout& operator=(const dropout& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_keep_prob = other.m_keep_prob;
    m_mask = other.m_mask
               ? std::unique_ptr<AbsDistMatrixType>(other.m_mask->Copy())
               : nullptr;
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc = other.m_tensors_dnn_desc;
    m_tensors_dnn_desc.set_layer(this);
    m_states = other.m_states;
    m_reserve_space = other.m_reserve_space;
    if (other.m_dropout_dnn_desc != nullptr) {
      setup_dropout_dnn_desc();
    }
#endif // LBANN_HAS_DNN_LIB
    return *this;
  }

  ~dropout() override = default;

  dropout* copy() const override { return new dropout(*this); }
  std::string get_type() const override { return "dropout"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("Keep probability", m_keep_prob);
    return desc;
  }
  /** @brief get prob for keep each unit. */
  EvalType get_keep_prob() const { return m_keep_prob; }
  /** @brief set prob for keep each unit. */
  void set_keep_prob(EvalType keep_prob) { m_keep_prob = keep_prob; }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims(DataReaderMetaData& dr_metadata) override
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());
  }

  void setup_data(size_t max_mini_batch_size) override
  {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
    m_mask = std::unique_ptr<AbsDistMatrixType>(this->get_activations().Copy());
  }

  void setup_gpu() override
  {
    data_type_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else

#ifdef LBANN_DETERMINISTIC
    /// @todo GPU implementation of dropout with sequential consistency
    if (this->get_comm()->am_trainer_master()) {
      LBANN_WARNING(this->get_type(),
                    " layer \"",
                    this->get_name(),
                    "\" ",
                    "does not guarantee sequential consistency");
    }
#endif // LBANN_DETERMINISTIC

    // Initialize DNN library objects
    setup_dropout_dnn_desc();

#endif // LBANN_HAS_DNN_LIB
  }

  void fp_compute() override
  {
    if (this->using_gpus()) {
      fp_compute_gpu();
    }
    else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override
  {
    if (this->using_gpus()) {
      bp_compute_gpu();
    }
    else {
      bp_compute_cpu();
    }
  }

private:
  void fp_compute_cpu();

  /** Adjust gradients for dropout in backprop. */
  void bp_compute_cpu();

  void fp_compute_gpu();

  void bp_compute_gpu();

#ifdef LBANN_HAS_DNN_LIB
  /** Setup DNN library dropout descriptor and RNG state.
   */
  void setup_dropout_dnn_desc()
  {

    // Setup RNG state
    size_t size = dnn_lib::get_dropout_states_size();
    m_states.Resize((size + sizeof(TensorDataType) - 1) /
                      sizeof(TensorDataType),
                    1);

    // Setup dropout descriptor
    m_dropout_dnn_desc.set(float(1 - m_keep_prob),
                           m_states.Buffer(),
                           m_states.Height() * sizeof(TensorDataType),
                           get_generator()());
  }
#endif // LBANN_HAS_DNN_LIB

  /** Probability of keeping each unit. */
  EvalType m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  std::unique_ptr<AbsDistMatrixType> m_mask;

#ifdef LBANN_HAS_DNN_LIB
  /** Dropout DNN library descriptor. */
  dnn_lib::DropoutDescriptor m_dropout_dnn_desc;
  /** Tensor DNN library descriptors. */
  dnn_lib::entrywise_layer_tensor_manager<TensorDataType> m_tensors_dnn_desc;
  /** RNG state for DNN library dropout. */
  El::Matrix<TensorDataType, El::Device::GPU> m_states;
  /** Work space for DNN library dropout. */
  El::Matrix<TensorDataType, El::Device::GPU> m_reserve_space;
#endif // LBANN_HAS_DNN_LIB
};

template <typename T, data_layout L, El::Device D>
using dropout_layer = dropout<T, L, D>;

LBANN_DEFINE_LAYER_BUILDER(dropout);

#ifndef LBANN_DROPOUT_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class dropout<T, data_layout::DATA_PARALLEL, Device>;        \
  extern template class dropout<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_DROPOUT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
