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

#ifndef LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

template <data_layout T_layout, El::Device Dev>
class cross_entropy_layer : public Layer {
public:

  cross_entropy_layer(lbann_comm *comm) : Layer(comm) {
    // Expects inputs for prediction and ground truth
    m_expected_num_parent_layers = 2;
  }

  cross_entropy_layer* copy() const override { return new cross_entropy_layer(*this); }
  std::string get_type() const override { return "cross entropy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    Layer::setup_dims();
    this->m_num_neurons = 1;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims = {1};
  }

  void fp_compute() override {

    // Initialize workspace
    std::unique_ptr<AbsDistMat> workspace;
    const auto& prediction = get_prev_activations(0);
    switch (get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      workspace.reset(new StarVCMat<Dev>(prediction.Grid(),
                                         prediction.Root()));
      break;
    case data_layout::MODEL_PARALLEL:
      workspace.reset(new StarMRMat<Dev>(prediction.Grid(),
                                         prediction.Root()));
      break;
    default: LBANN_ERROR("invalid data layout");
    }
    workspace->AlignWith(prediction.DistData());
#ifdef HYDROGEN_HAVE_CUB
    if (workspace->GetLocalDevice() == El::Device::GPU) {
      workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
    El::Zeros(*workspace, 1, prediction.Width());

    // Compute local contributions and accumulate
    /// @todo Consider reduce rather than allreduce
    local_fp_compute(get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     workspace->Matrix());
    m_comm->allreduce(*workspace, workspace->RedundantComm());
    El::Copy(*workspace, get_activations());
    
  }
  
  void bp_compute() override {

    // Initialize workspace
    std::unique_ptr<AbsDistMat> workspace;
    const auto& prediction = get_prev_activations(0);
    switch (get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      workspace.reset(new StarVCMat<Dev>(prediction.Grid(),
                                         prediction.Root()));
      break;
    case data_layout::MODEL_PARALLEL:
      workspace.reset(new StarMRMat<Dev>(prediction.Grid(),
                                         prediction.Root()));
      break;
    default: LBANN_ERROR("invalid data layout");
    }
    workspace->AlignWith(prediction.DistData());
#ifdef HYDROGEN_HAVE_CUB
    if (workspace->GetLocalDevice() == El::Device::GPU) {
      workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
    El::Copy(get_prev_error_signals(), *workspace);

    // Compute local gradients
    local_bp_compute(get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     workspace->LockedMatrix(),
                     get_local_error_signals(0),
                     get_local_error_signals(1));
    
  }

private:

  /** Compute local contributions to cross entropy loss. */
  static void local_fp_compute(const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               AbsMat& local_contribution);
  /** Compute local gradients. */
  static void local_bp_compute(const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               const AbsMat& local_gradient_wrt_output,
                               AbsMat& local_gradient_wrt_prediction,
                               AbsMat& local_gradient_wrt_ground_truth);
  
};

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
