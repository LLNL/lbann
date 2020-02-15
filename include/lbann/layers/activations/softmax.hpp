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

#ifndef LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/distconv.hpp"

// Threshold outputs to a minimum value.
// If enabled, the minimum output value is sqrt(min), where min is the
// minimum, normalized, positive value (~1e-19 for float and ~1e-154
// for double). The gradients w.r.t. input will be inaccurate, on the
// order of the minimum output value.
#define LBANN_ENABLE_SOFTMAX_CUTOFF

namespace lbann {

enum class softmax_mode {INVALID, INSTANCE, CHANNEL};

/** @brief
 *
 *  @f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 */
template <data_layout Layout, El::Device Device>
class softmax_layer : public Layer {
public:

  softmax_layer(lbann_comm *comm,
                softmax_mode mode)
    : Layer(comm), m_mode(mode)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {
    if(mode == softmax_mode::INVALID) {
      LBANN_ERROR("invalid softmax mode");
    }
  }

  softmax_layer(const softmax_layer& other)
    : Layer(other), m_mode(other.m_mode),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  softmax_layer& operator=(const softmax_layer& other) {
    Layer::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() : nullptr);
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~softmax_layer() = default;

  softmax_layer* copy() const override { return new softmax_layer(*this); }
  std::string get_type() const override { return "softmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims(get_input_dims());
  }

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    auto dist = get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMat::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    Layer::fp_setup_outputs(mini_batch_size);
    const auto& dist_data = get_prev_activations().DistData();
    m_workspace->Empty(false);
    m_workspace->AlignWith(dist_data);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Softmax mode. */
  const softmax_mode m_mode;

  /** Workspace for column-wise reductions. */
  std::unique_ptr<AbsDistMat> m_workspace;

#ifdef LBANN_HAS_CUDNN
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

#ifdef LBANN_HAS_DISTCONV
 protected:
  dc::Softmax *m_softmax;

  void fp_compute_distconv();
  void bp_compute_distconv();

 public:
  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(dists, invariants, updated, fixed);
    if (!this->distconv_enabled()) return;

    // No overlap supported yet
    const dc::IntVector no_overlap(dc::num_dims, 0);
    for (int i = 0; i < 4; ++i) {
      auto &dist = dists[this][i];
      dist.set_overlap(no_overlap);
      updated.insert(&dist);
      fixed.insert(&dist);
    }
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    Layer::setup_tensors_fwd(dists);
    if (!distconv_enabled()) return;
    setup_prev_activations_tensor(dists);
    setup_activations_tensor(dists);
    setup_activations_copyout_tensor(dists);
  }
  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    Layer::setup_tensors_bwd(dists);
    if (!distconv_enabled()) return;
    setup_prev_error_signals_tensor(dists);
    setup_error_signals_tensor(dists);
    setup_error_signals_copyout_tensor(dists);
    m_softmax = new dc::Softmax(dc::get_backend());
    auto dc_softmax_mode = m_mode == softmax_mode::INSTANCE ?
        ::distconv::SoftmaxMode::INSTANCE : ::distconv::SoftmaxMode::CHANNEL;
    m_softmax->setup(m_prev_activations_t, dc_softmax_mode);
  }

#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_SOFTMAX_LAYER_INSTANTIATE
extern template class softmax_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class softmax_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class softmax_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class softmax_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
