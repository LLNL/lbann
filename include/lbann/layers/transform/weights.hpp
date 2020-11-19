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

#ifndef LBANN_LAYER_WEIGHTS_HPP_INCLUDED
#define LBANN_LAYER_WEIGHTS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

/** @brief Output a weights tensor.
 *
 *  Interfaces with a @c weights object and outputs its tensor.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class weights_layer : public data_type_layer<TensorDataType> {

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The local tensor type expected in this object. */
  using AbsMatrixType = El::AbstractMatrix<TensorDataType>;

  /** @brief The device-specific local tensor type. */
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

#ifdef LBANN_HAS_GPU
  /** @brief The GPU device-specific local tensor type. */
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;
#endif

  /** @brief The concrete optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

 public:
  weights_layer(lbann_comm *comm, std::vector<El::Int> dims)
    : data_type_layer<TensorDataType>(comm) {
    std::vector<int> dims_;
    for (const auto& d : dims) { dims_.push_back(d); }
    this->set_output_dims(dims_);
    this->m_expected_num_parent_layers = 0;
  }

  weights_layer(const weights_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_gradient(other.m_gradient ? other.m_gradient->Copy() : nullptr) {
    m_workspace.SetMemoryMode(other.m_workspace.MemoryMode());
  }
  weights_layer& operator=(const weights_layer& other){
    data_type_layer<TensorDataType>::operator=(other);
    m_gradient.reset(other.m_gradient ? other.m_gradient->Copy() : nullptr);
    m_workspace.SetMemoryMode(other.m_workspace.MemoryMode());
    return *this;
  }
  weights_layer* copy() const override { return new weights_layer(*this); }
  std::string get_type() const override { return "weights"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_matrices(const El::Grid& grid) override {
    data_type_layer<TensorDataType>::setup_matrices(grid);

    // Initialize weights gradient
    auto dist = this->get_activations().DistData();
    dist.rowDist = El::STAR;
    m_gradient.reset(AbsDistMatrixType::Instantiate(dist));

    // Initialize workspace
#if defined HYDROGEN_HAVE_CUB
    if (Dev == El::Device::GPU)
      m_workspace.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // defined HYDROGEN_HAVE_CUB
  }

  void setup_data(size_t max_mini_batch_size) override {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Initialize default weights if none are provided
    if (!this->has_weights()) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto init = make_unique<constant_initializer<DataType>>(DataType(0));
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_weights");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->add_weights(w);
      this->m_model->add_weights(std::move(w));
    }
    if (this->num_weights() != 1) {
      LBANN_ERROR("attempted to setup ",
                  this->get_type()," layer \"",this->get_name(),"\" ",
                  "with an invalid number of weights ",
                  "(expected at most 1, ",
                  "but found ",this->num_weights(),")");
    }

    // Setup weights and weights gradient
    m_gradient->AlignWith(this->get_activations());
    m_gradient->Resize(this->get_output_size(), 1);
    this->get_weights(0).set_dims(this->get_output_dims());
    this->get_weights(0).set_matrix_distribution(m_gradient->DistData());

    // Initialize freeze state
    if (this->m_frozen) { this->get_weights(0).freeze(); }
    else                { this->get_weights(0).unfreeze(); }
    if (this->get_weights(0).is_frozen() != this->m_frozen) {
      LBANN_ERROR((this->m_frozen ? "" : "un"),"frozen ",
                  "layer \"",this->get_name(),"\" has ",
                  (this->get_weights(0).is_frozen() ? "" : "un"),"frozen ",
                  "weights \"",this->get_weights(0).get_name(),"\"");
    }

  }

  void fp_compute() override {

    // Matrices
    const auto& local_weights = this->weights_values(0).LockedMatrix();
    auto& local_output = this->get_local_activations();
    El::Ones(m_workspace, local_output.Width(), 1);

    // Duplicate weights across matrix columns
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             El::TypeTraits<TensorDataType>::One(), local_weights, m_workspace,
             El::TypeTraits<TensorDataType>::Zero(), local_output);

    // Clean up
    m_workspace.Empty();

  }

  void bp_compute() override {

    // Get optimizer
    // Note: Nothing needs to be done if there is no optimizer
    auto* opt = this->get_weights(0).get_optimizer();
    if (opt == nullptr) { return; }

    // Matrices
    const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
    El::Ones(m_workspace, local_gradient_wrt_output.Width(), 1);

    El::Gemv(El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_gradient_wrt_output, m_workspace,
             El::TypeTraits<TensorDataType>::Zero(),
             m_gradient->Matrix());

    opt->add_to_gradient(*m_gradient,
                         El::TypeTraits<TensorDataType>::One(),
                         true);

    // Clean up
    m_workspace.Empty();

  }

 private:

  /** Weights gradient. */
  std::unique_ptr<AbsDistMatrixType> m_gradient;
  /** Workspace. */
  El::Matrix<TensorDataType, Dev> m_workspace;
};

LBANN_DEFINE_LAYER_BUILDER(weights);

#ifndef LBANN_WEIGHTS_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class weights_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class weights_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_WEIGHTS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_WEIGHTS_HPP_INCLUDED
