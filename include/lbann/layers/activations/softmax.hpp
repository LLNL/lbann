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

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/distconv.hpp"

// Threshold outputs to a minimum value.

// If enabled, the minimum output value is sqrt(min), where min is the
// minimum, normalized, positive value (~1e-19 for float and ~1e-154
// for double). During backprop, gradients are computed as if
// thresholding did not occur, so there will be a discrepancy for
// values that are thresholded.
#define LBANN_ENABLE_SOFTMAX_THRESHOLD

namespace lbann {

enum class softmax_mode {INVALID, INSTANCE, CHANNEL};

/**
>>>>>>> bc1c6f45a8e0afa896c6961bd0a28ee19dfd3f82
 *  @f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class softmax_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  softmax_layer(lbann_comm *comm,
                softmax_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_mode(mode)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {
    if(mode == softmax_mode::INVALID) {
      LBANN_ERROR("invalid softmax mode");
    }
  }

  softmax_layer(const softmax_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_mode(other.m_mode),
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

  ~softmax_layer() = default;

  softmax_layer* copy() const override { return new softmax_layer(*this); }
  std::string get_type() const override { return "softmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }

  void setup_matrices(const El::Grid& grid) override {
    data_type_layer<TensorDataType>::setup_matrices(grid);
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMatrixType::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
    const auto& dist_data = this->get_prev_activations().DistData();
    m_workspace->Empty(false);
    m_workspace->AlignWith(dist_data);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

  template <typename U>
  friend void fp_compute_impl(softmax_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(softmax_layer<U, Layout, Device>& l);

private:

  /** Softmax mode. */
  const softmax_mode m_mode;

  /** Workspace for column-wise reductions. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_CUDNN
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager<TensorDataType> m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

// Minimum output value to avoid denormalized floats
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
  const TensorDataType threshold_val = static_cast<TensorDataType>(El::Sqrt(std::numeric_limits<TensorDataType>::min()));
#else
  const TensorDataType threshold_val = El::TypeTraits<TensorDataType>::Zero();
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD

#ifdef LBANN_HAS_DISTCONV
 protected:
  dc::Softmax *m_softmax;

  void fp_compute_distconv();
  void bp_compute_distconv();

 public:
  void init_distribution(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    data_type_layer<TensorDataType>::init_distribution(
        dists, invariants, updated, fixed);
    if (!this->distconv_enabled()) return;

    // No overlap supported yet
    const dc::IntVector no_overlap(this->get_num_dims(), 0);
    for (int i = 0; i < 4; ++i) {
      auto &dist = dists[this][i];
      dist.set_overlap(no_overlap);
      updated.insert(&dist);
      fixed.insert(&dist);
    }
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    data_type_layer<TensorDataType>::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;
    this->setup_prev_activations_tensor(dists);
    this->setup_activations_tensor(dists);
    this->setup_activations_copyout_tensor(dists);
  }
  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    data_type_layer<TensorDataType>::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;
    this->setup_prev_error_signals_tensor(dists);
    this->setup_error_signals_tensor(dists);
    this->setup_error_signals_copyout_tensor(dists);
    m_softmax = new dc::Softmax(dc::get_backend());
    auto dc_softmax_mode = m_mode == softmax_mode::INSTANCE ?
        ::distconv::SoftmaxMode::INSTANCE : ::distconv::SoftmaxMode::CHANNEL;
    m_softmax->setup(this->get_prev_activations_t(), dc_softmax_mode);
  }
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_SOFTMAX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class softmax_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class softmax_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
