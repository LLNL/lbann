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

#ifndef LBANN_LAYERS_LOSS_L2_NORM2_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_L2_NORM2_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Square of L2 vector norm.
 *
 *  @f[ \lVert x\rVert_2^2 = \sum\limits_{i} x_i^2 @f]
 */
template <data_layout T_layout, El::Device Dev>
class l2_norm2_layer : public Layer {
public:

  l2_norm2_layer(lbann_comm *comm) : Layer(comm) {}

  l2_norm2_layer(const l2_norm2_layer& other)
    : Layer(other),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr) {}
  l2_norm2_layer& operator=(const l2_norm2_layer& other) {
    Layer::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() : nullptr);
    return *this;
  }

  l2_norm2_layer* copy() const override { return new l2_norm2_layer(*this); }
  std::string get_type() const override { return "L2 norm squared"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});
  }

  void setup_data() override {
    Layer::setup_data();

    // Initialize workspace
    auto dist = get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMat::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB

  }

  void fp_compute() override {

    // Initialize workspace
    m_workspace->Empty();
    m_workspace->AlignWith(get_prev_activations());
    m_workspace->Resize(1, get_prev_activations().Width());

    // Compute local contributions and accumulate
    /// @todo Consider reduce rather than allreduce
    local_fp_compute(get_local_prev_activations(),
                     m_workspace->Matrix());
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());
    El::Copy(*m_workspace, get_activations());

    // Clean up
    m_workspace->Empty();

  }

  void bp_compute() override {

    // Initialize workspace
    m_workspace->Empty();
    m_workspace->AlignWith(get_prev_activations());
    El::Copy(get_prev_error_signals(), *m_workspace);

    // Compute local gradients
    local_bp_compute(get_local_prev_activations(),
                     m_workspace->LockedMatrix(),
                     get_local_error_signals());

    // Clean up
    m_workspace->Empty();

  }

private:

  /** Compute local contributions to L2 norm. */
  static void local_fp_compute(const AbsMat& local_input,
                               AbsMat& local_contribution);
  /** Compute local gradients. */
  static void local_bp_compute(const AbsMat& local_input,
                               const AbsMat& local_gradient_wrt_output,
                               AbsMat& local_gradient_wrt_input);

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMat> m_workspace;

};

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_L2_NORM2_HPP_INCLUDED
