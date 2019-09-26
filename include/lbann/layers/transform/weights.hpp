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

#include "lbann/layers/transform/transform.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

/** @brief Output a weights tensor.
 *
 *  Interfaces with a @c weights object and outputs its tensor.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class weights_layer : public transform_layer<TensorDataType> {

 public:
  weights_layer(lbann_comm *comm, std::vector<El::Int> dims)
    : transform_layer<TensorDataType>(comm) {
    std::vector<int> dims_;
    for (const auto& d : dims) { dims_.push_back(d); }
    this->set_output_dims(dims_);
    this->m_expected_num_parent_layers = 0;
  }

  weights_layer(const weights_layer& other)
    : transform_layer<TensorDataType>(other),
      m_gradient(other.m_gradient ? other.m_gradient->Copy() : nullptr) {
    if (other.m_workspace) {
      switch (other.m_workspace->GetDevice()) {
      case El::Device::CPU: m_workspace.reset(new El::Matrix<TensorDataType, El::Device::CPU>()); break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU: m_workspace.reset(new El::Matrix<TensorDataType, El::Device::GPU>()); break;
#endif // LBANN_HAS_GPU
      default: LBANN_ERROR("unknown device type");
      }
      m_workspace->SetMemoryMode(other.m_workspace->MemoryMode());
    }
  }
  weights_layer& operator=(const weights_layer& other){
    transform_layer<TensorDataType>::operator=(other);
    m_gradient.reset(other.m_gradient ? other.m_gradient->Copy() : nullptr);
    m_workspace.reset();
    if (other.m_workspace) {
      switch (other.m_workspace->GetDevice()) {
      case El::Device::CPU: m_workspace.reset(new El::Matrix<TensorDataType, El::Device::CPU>()); break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU: m_workspace.reset(new El::Matrix<TensorDataType, El::Device::GPU>()); break;
#endif // LBANN_HAS_GPU
      default: LBANN_ERROR("unknown device type");
      }
      m_workspace->SetMemoryMode(other.m_workspace->MemoryMode());
    }
    return *this;
  }
  weights_layer* copy() const override { return new weights_layer(*this); }
  std::string get_type() const override { return "weights"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_matrices(const El::Grid& grid) override {
    transform_layer<TensorDataType>::setup_matrices(grid);

    // Initialize weights gradient
    auto dist = this->get_activations().DistData();
    dist.rowDist = El::STAR;
    m_gradient.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(dist));

    // Initialize workspace
    switch (Dev) {
    case El::Device::CPU: m_workspace.reset(new El::Matrix<TensorDataType, El::Device::CPU>()); break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_workspace.reset(new El::Matrix<TensorDataType, El::Device::GPU>());
#ifdef HYDROGEN_HAVE_CUB
      m_workspace->SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
      break;
#endif // LBANN_HAS_GPU
    default: LBANN_ERROR("unknown device type");
    }

  }

  void setup_data() override {
    transform_layer<TensorDataType>::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.empty()) {
      auto w = make_unique<weights>(get_comm());
      auto init = make_unique<constant_initializer>(DataType(0));
      std::unique_ptr<optimizer<TensorDataType>> opt(m_model->create_optimizer());
      w->set_name(get_name() + "_weights");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->m_weights.push_back(w.get());
      this->m_model->add_weights(std::move(w));
    }
    if (this->m_weights.size() != 1) {
      LBANN_ERROR("attempted to setup ",
                  get_type()," layer \"",get_name(),"\" ",
                  "with an invalid number of weights ",
                  "(expected at most 1, ",
                  "but found ",this->m_weights.size(),")");
    }

    // Setup weights and weights gradient
    m_gradient->AlignWith(get_activations());
    m_gradient->Resize(get_output_size(), 1);
    m_weights[0]->set_dims(get_output_dims());
    m_weights[0]->set_matrix_distribution(m_gradient->DistData());

    // Initialize freeze state
    if (this->m_frozen) { m_weights[0]->freeze(); }
    else                { m_weights[0]->unfreeze(); }
    if (m_weights[0]->is_frozen() != this->m_frozen) {
      LBANN_ERROR((m_frozen ? "" : "un"),"frozen ",
                  "layer \"",get_name(),"\" has ",
                  (m_weights[0]->is_frozen() ? "" : "un"),"frozen ",
                  "weights \"",m_weights[0]->get_name(),"\"");
    }

  }

  void fp_compute() override {

    // Matrices
    const auto& local_weights = m_weights[0]->get_values().LockedMatrix();
    auto& local_output = get_local_activations();
    m_workspace->Resize(local_output.Width(), 1);
    El::Fill(*m_workspace, DataType(1));

    // Duplicate weights across matrix columns
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             DataType(1), local_weights, *m_workspace,
             DataType(0), local_output);

    // Clean up
    m_workspace->Empty();

  }

  void bp_compute() override {

    // Get optimizer
    // Note: Nothing needs to be done if there is no optimizer
    auto* opt = this->m_weights[0]->get_optimizer();
    if (opt == nullptr) { return; }

    // Matrices
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    m_workspace->Resize(local_gradient_wrt_output.Width(), 1);
    El::Fill(*m_workspace, DataType{1});

    El::Gemv(El::NORMAL,
             DataType{1}, local_gradient_wrt_output, *m_workspace,
             DataType{0}, m_gradient->Matrix());
    opt->add_to_gradient(*m_gradient, DataType{1}, true);

    // Clean up
    m_workspace->Empty();

  }

 private:

  /** Weights gradient. */
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> m_gradient;
  /** Workspace. */
  std::unique_ptr<El::AbstractMatrix<TensorDataType>> m_workspace;

};

#ifndef LBANN_WEIGHTS_LAYER_INSTANTIATE
extern template class weights_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class weights_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class weights_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class weights_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_WEIGHTS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_WEIGHTS_HPP_INCLUDED
