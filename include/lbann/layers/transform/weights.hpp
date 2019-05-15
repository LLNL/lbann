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

namespace lbann {

/** @brief Output a weights tensor.
 *
 *  Interfaces with a @c weights object and outputs its tensor.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class weights_layer : public transform_layer {

 public:
  weights_layer(lbann_comm *comm, std::vector<El::Int> dims)
    : transform_layer(comm) {
    std::vector<int> dims_;
    for (const auto& d : dims) { dims_.push_back(d); }
    set_output_dims(dims_);
    this->m_expected_num_parent_layers = 0;
  }

  weights_layer(const weights_layer& other)
    : transform_layer(other),
      m_gradient(other.m_gradient ? other.m_gradient->Copy() : nullptr) {
    if (other.m_workspace) {
      switch (other.m_workspace->GetDevice()) {
      case El::Device::CPU: m_workspace.reset(new CPUMat()); break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU: m_workspace.reset(new GPUMat()); break;
#endif // LBANN_HAS_GPU
      default: LBANN_ERROR("unknown device type");
      }
      m_workspace->SetMemoryMode(other.m_workspace->MemoryMode());
    }
  }
  weights_layer& operator=(const weights_layer& other){
    transform_layer::operator=(other);
    m_gradient.reset(other.m_gradient ? other.m_gradient->Copy() : nullptr);
    m_workspace.reset();
    if (other.m_workspace) {
      switch (other.m_workspace->GetDevice()) {
      case El::Device::CPU: m_workspace.reset(new CPUMat()); break;
#ifdef LBANN_HAS_GPU
      case El::Device::GPU: m_workspace.reset(new GPUMat()); break;
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
    transform_layer::setup_matrices(grid);

    // Initialize weights gradient
    auto dist = get_activations().DistData();
    dist.rowDist = El::STAR;
    m_gradient.reset(AbsDistMat::Instantiate(dist));

    // Initialize workspace
    switch (Dev) {
    case El::Device::CPU: m_workspace.reset(new CPUMat()); break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      m_workspace.reset(new GPUMat());
#ifdef HYDROGEN_HAVE_CUB
      m_workspace->SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
      break;
#endif // LBANN_HAS_GPU
    default: LBANN_ERROR("unknown device type");
    }

  }

  void setup_data() override {
    transform_layer::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 1) {
      std::stringstream err;
      err << "attempted to setup "
          << get_type() << " layer \"" << get_name() << "\" "
          << "with an invalid number of weights "
          << "(expected at most 1, "
          << "but found " << this->m_weights.size() << ")";
      LBANN_ERROR(err.str());
    }
    this->m_weights.resize(1, nullptr);
    auto& w = this->m_weights[0];
    if (w == nullptr) {
      w = new weights(get_comm());
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(0)));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      w->set_name(get_name() + "_weights");
      w->set_initializer(init);
      w->set_optimizer(opt);
      this->m_model->add_weights(w);
    }

    // Setup weights and weights gradient
    m_gradient->AlignWith(get_activations());
    m_gradient->Resize(get_output_size(), 1);
    w->set_dims(get_output_dims());
    w->set_matrix_distribution(m_gradient->DistData());

    // Initialize freeze state
    if (this->m_frozen) { w->freeze(); }
    else                { w->unfreeze(); }
    if (w->is_frozen() != this->m_frozen) {
      std::stringstream err;
      err << (m_frozen ? "" : "un") << "frozen "
          << "layer \"" << get_name() << "\" has "
          << (w->is_frozen() ? "" : "un") << "frozen "
          << "weights \"" << w->get_name() << "\"";
      LBANN_ERROR(err.str());
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
    constexpr DataType zero = 0;
    constexpr DataType one = 1;

    // Get optimizer
    // Note: Nothing needs to be done if there is no optimizer
    auto* opt = this->m_weights[0]->get_optimizer();
    if (opt == nullptr) { return; }

    // Matrices
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    m_workspace->Resize(local_gradient_wrt_output.Width(), 1);
    El::Fill(*m_workspace, one);

    // Compute gradient contribution and accumulate
    const auto& scale = one / this->m_model->get_effective_mini_batch_size();
    El::Gemv(El::NORMAL,
             scale, local_gradient_wrt_output, *m_workspace,
             zero, m_gradient->Matrix());
    opt->add_to_gradient(*m_gradient, one, true);

    // Clean up
    m_workspace->Empty();

  }

 private:

  /** Weights gradient. */
  std::unique_ptr<AbsDistMat> m_gradient;
  /** Workspace. */
  std::unique_ptr<AbsMat> m_workspace;

};

} // namespace lbann

#endif // LBANN_LAYER_WEIGHTS_HPP_INCLUDED
