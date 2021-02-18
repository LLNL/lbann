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

#define LBANN_DATA_TYPE_LAYER_INSTANTIATE

#include "matrix_builder.hpp"

#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/summary_impl.hpp"

namespace lbann {

template <typename TensorDataType>
data_type_layer<TensorDataType>::data_type_layer(const data_type_layer<TensorDataType>& other) :
  Layer(other),
  m_persistent_error_signals(other.m_persistent_error_signals) {

  // Deep matrix copies
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
}

template <typename TensorDataType>
data_type_layer<TensorDataType>& data_type_layer<TensorDataType>::operator=(const data_type_layer<TensorDataType>& other) {
  Layer::operator=(other);

  // Deep matrix copies
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? ptr->Copy() : nullptr);
  }
  m_persistent_error_signals = other.m_persistent_error_signals;
  return *this;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::forward_prop_sample(El::Matrix<float, El::device::CPU> samples) {
  // Setup weights proxies
  if (this->has_weights()) {
    if ((m_weights_proxy.size() == 0) || m_weights_proxy[0].empty()) {
      auto const num_weights = this->num_weights();
      m_weights_proxy.resize(num_weights);
      const auto ptrs = this->get_weights_pointers();
      for (size_t ii = 0; ii < num_weights; ++ii) {
        m_weights_proxy[ii].setup(ptrs[ii]);
      }
    }
    for (auto& wp : m_weights_proxy)
      wp.synchronize_with_master();
  }

  // Setup tensors
  int mini_batch_size = 64;
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { hydrogen::gpu::SynchronizeDevice(); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  // Apply layer's compute function
  fp_sample(samples);

  // Add this layer as a gradient source for weight optimizers
  this->add_as_gradient_source();

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { hydrogen::gpu::SynchronizeDevice(); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::forward_prop() {
  const auto fp_start = get_time();

  // Setup weights proxies
  if (this->has_weights()) {
    if ((m_weights_proxy.size() == 0) || m_weights_proxy[0].empty()) {
      auto const num_weights = this->num_weights();
      m_weights_proxy.resize(num_weights);
      const auto ptrs = this->get_weights_pointers();
      for (size_t ii = 0; ii < num_weights; ++ii) {
        m_weights_proxy[ii].setup(ptrs[ii]);
      }
    }
    for (auto& wp : m_weights_proxy)
      wp.synchronize_with_master();
  }

  // Setup tensors
  const auto& c = static_cast<sgd_execution_context&>(m_model->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { hydrogen::gpu::SynchronizeDevice(); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) get_distconv_adapter().fp_setup(mini_batch_size);
#endif // LBANN_HAS_DISTCONV

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) get_distconv_adapter().fp_postprocess();
#endif // LBANN_HAS_DISTCONV

  // Add this layer as a gradient source for weight optimizers
  this->add_as_gradient_source();

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { hydrogen::gpu::SynchronizeDevice(); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_fp_time += get_time() - fp_start;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::back_prop_impl_() {
  const auto bp_start = get_time();

  // Setup tensors
  const auto& c = static_cast<sgd_execution_context&>(
    m_model->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  bp_setup_gradient_wrt_inputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { hydrogen::gpu::SynchronizeDevice(); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) get_distconv_adapter().bp_setup(mini_batch_size);
#endif // LBANN_HAS_DISTCONV

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) get_distconv_adapter().bp_postprocess();
#endif // LBANN_HAS_DISTCONV

  // Remove this layer as a gradient source for weight optimizers
  this->remove_as_gradient_source();

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { hydrogen::gpu::SynchronizeDevice(); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_bp_time += get_time() - bp_start;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::summarize_matrices(lbann_summary& summarizer, int step) {

  // Summarize activation matrices
  const int num_children = get_num_children();
  for (int i = 0; i < num_children; ++i) {
    AbsDistMatReadProxyType<El::Device::CPU> acts(*m_outputs[i]);
    std::string prefix = m_name + "/activations";
    if (num_children > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", acts.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", acts.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", acts.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", acts.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", acts.GetLocked(), step);
  }

  // Summarize error signal matrices
  const int num_parents = get_num_parents();
  for (int i = 0; i < num_parents; ++i) {
    if (!m_gradient_wrt_inputs[i]) continue;

    AbsDistMatReadProxyType<El::Device::CPU> error_signals(*m_gradient_wrt_inputs[i]);
    std::string prefix = m_name + "/error_signals";
    if (num_parents > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", error_signals.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", error_signals.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", error_signals.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", error_signals.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", error_signals.GetLocked(), step);
  }

}

// ===================================================================
// Tensor access functions
// ===================================================================

// Accessing distributed matrices
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_prev_activations(int parent_index) const -> const AbsDistMatrixType& {
  if (parent_index < 0 || parent_index >= (int) m_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_inputs.size() << " previous activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_inputs[parent_index];
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_activations(int child_index) const -> const AbsDistMatrixType& {
  if (child_index < 0 || child_index >= (int) m_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_outputs.size() << " activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_outputs[child_index];
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_prev_error_signals(int child_index) const -> const AbsDistMatrixType& {
  if (child_index < 0 || child_index >= (int) m_gradient_wrt_outputs.size()) {
    LBANN_ERROR(
      "Attempted to access invalid previous error signal matrix "
      "from ", m_name, ".\n\nRequested index ", child_index, ", "
      "but there are ", m_gradient_wrt_outputs.size(),
      " previous error signal matrices)");
  }
  if (!m_gradient_wrt_outputs[child_index]) {
    LBANN_ERROR("Previous error signal from", m_name,
                "(index=", child_index, ") is not currently allocated.");
  }
  return *m_gradient_wrt_outputs[child_index];
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_error_signals(int parent_index) const
  -> const AbsDistMatrixType& {
  if (parent_index < 0 || parent_index >= (int) m_gradient_wrt_inputs.size()) {
    LBANN_ERROR("Attempted to access invalid error signal matrix "
                "from ", m_name, ". Requested index ", parent_index, ", "
                "but there are ", m_gradient_wrt_inputs.size(),
                " error signal matrices)");
  }
  if (!m_gradient_wrt_inputs[parent_index]) {
    LBANN_ERROR("Error signal ", parent_index,
                " is currently not available.\n",
                "num parents = ", get_num_parents(), "\n",
                "num children = ", get_num_children(), "\n");
  }
  return *m_gradient_wrt_inputs[parent_index];
}

// Accessing non-const distributed matrices
// Note: Using idiom from Item 3, p. 23 in "Effective C++", 3rd ed.,
// by Scott Meyers.
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_activations(int child_index) -> AbsDistMatrixType& {
  return const_cast<AbsDistMatrixType&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_activations(child_index));
}

template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_error_signals(int parent_index) -> AbsDistMatrixType& {
  return const_cast<AbsDistMatrixType&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_error_signals(parent_index));
}

// Accessing local matrices
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_activations(int child_index) -> AbsMatrixType& {
  return get_activations(child_index).Matrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) -> AbsMatrixType& {
  return get_error_signals(parent_index).Matrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_prev_activations(int parent_index) const -> const AbsMatrixType&{
  return get_prev_activations(parent_index).LockedMatrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_activations(int child_index) const -> const AbsMatrixType& {
  return get_activations(child_index).LockedMatrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_prev_error_signals(int child_index) const -> const AbsMatrixType& {
  return get_prev_error_signals(child_index).LockedMatrix();
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) const -> const AbsMatrixType& {
  return get_error_signals(parent_index).LockedMatrix();
}

// Accessing matrices corresponding to parent/child layer
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_activations(const Layer& child) const -> const BaseDistMat& {
  if (this->get_num_children() <= 0) {
    LBANN_ERROR("This layer has no children");
  }
  const int child_index = find_child_layer_index(child);
  if (child_index >= get_num_children()) {
    std::stringstream err;
    err << "attempted to get activation tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << child.get_name() << "\", "
        << "which is not a child layer";
    LBANN_ERROR(err.str());
  }
  return get_activations(child_index);
}
template <typename TensorDataType>
auto data_type_layer<TensorDataType>::get_error_signals(const Layer& parent) const -> const BaseDistMat& {
  const int parent_index = find_parent_layer_index(parent);
  if (parent_index >= get_num_parents()) {
    LBANN_ERROR("attempted to get error signal tensor of "
                "layer \"", get_name(), "\" "
                "corresponding to layer\"", parent.get_name(), "\", "
                "which is not a parent layer");
  }
  return get_error_signals(parent_index);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::set_keep_error_signals(bool flag)
{
  m_persistent_error_signals = flag;
}

namespace {

// Some indirection around building matrices to keep things tidy in
// the real code. This is just to hide multiple switches without
// building a full-blown dispatch engine... This also keeps bad
// type/device combinations from being instantiated (eg, cpu_fp16 on
// Device::GPU).
using namespace h2::meta;

#ifdef LBANN_HAS_GPU
template <typename T, data_layout Layout,
          typename=EnableWhenV<El::IsStorageType<T, El::Device::GPU>>>
auto MakeMatBuilderGPU()
  -> std::unique_ptr<details::MatrixBuilder<T>>
{
  return make_unique<
      details::DefaultMemoryMatrixBuilder<T,Layout,El::Device::GPU>>();
}

template <typename T, data_layout Layout,
          typename=EnableUnlessV<El::IsComputeType<T, El::Device::GPU>>,
          typename=void>
auto MakeMatBuilderGPU()
  -> std::unique_ptr<details::MatrixBuilder<T>>
{
  LBANN_ERROR("Bad type/device combination.");
  return nullptr;
}
#endif // LBANN_HAS_GPU

template <typename T, data_layout Layout>
auto MakeMatBuilderDev(El::Device const device)
  -> std::unique_ptr<details::MatrixBuilder<T>>
{
  switch (device) {
  case El::Device::CPU:
    return make_unique<
      details::DefaultMemoryMatrixBuilder<T,Layout,El::Device::CPU>>();
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return MakeMatBuilderGPU<T, Layout>();
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Invalid device type");
  }
}
template <typename T>
auto MakeMatBuilder(data_layout const layout, El::Device const device)
  -> std::unique_ptr<details::MatrixBuilder<T>>
{
  switch (layout) {
  case data_layout::DATA_PARALLEL:
    return MakeMatBuilderDev<T, data_layout::DATA_PARALLEL>(device);
  case data_layout::MODEL_PARALLEL:
    return MakeMatBuilderDev<T, data_layout::MODEL_PARALLEL>(device);
  default:
    LBANN_ERROR("Invalid data layout");
  }
  return nullptr;
}

}// namespace <anon>

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_matrices(const El::Grid& grid) {

  using MatrixBuilderType = details::MatrixBuilder<TensorDataType>;

  // DEBUG
  {
    char* keep_error_signals = getenv("LBANN_KEEP_ERROR_SIGNALS");
    if (!keep_error_signals || (std::stoi(keep_error_signals) == 0))
      m_persistent_error_signals = false;
    else
      m_persistent_error_signals = true;
  }

  // If no CUB, force persistent error signals:
#if defined(HYDROGEN_HAVE_GPU) && !defined(HYDROGEN_HAVE_CUB)
  if (this->get_device_allocation() == El::Device::GPU)
    m_persistent_error_signals = true;
#endif

  // Figure out how to make new matrices
  std::unique_ptr<MatrixBuilderType> mat_builder =
    MakeMatBuilder<TensorDataType>(
      this->get_data_layout(), this->get_device_allocation());

  // Destroy previously setup matrices
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();

  // Construct matrices
  m_inputs.resize(get_num_parents());
  m_outputs.resize(get_num_children());
  m_gradient_wrt_outputs.resize(get_num_children());
  m_gradient_wrt_inputs.resize(get_num_parents());
  for (auto& input : m_inputs) {
    input = mat_builder->MakeEmpty(grid, 0);
  }
  for (auto& output : m_outputs) {
    output = mat_builder->MakeEmpty(grid, 0);
  }
  for (auto& grad_wrt_input : m_gradient_wrt_inputs) {
    grad_wrt_input = mat_builder->MakeEmpty(grid, 0);
  }
  for (auto& grad_wrt_output : m_gradient_wrt_outputs) {
    grad_wrt_output = mat_builder->MakeEmpty(grid, 0);
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_data(size_t max_mini_batch_size) {
  Layer::setup_data(max_mini_batch_size);

  // Initialize input and output tensors
  fp_setup_inputs(max_mini_batch_size);
  fp_setup_outputs(max_mini_batch_size);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_compute() {
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zero(get_error_signals(i));
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::check_setup() {
  Layer::check_setup();
  std::stringstream err;

  // Check number of tensors
  const int num_parents = get_num_parents();
  const int num_children = get_num_children();
  if ((int) m_inputs.size() != num_parents
      || (int) m_outputs.size() != num_children
      || (int) m_gradient_wrt_outputs.size() != num_children
      || (int) m_gradient_wrt_inputs.size() != num_parents) {
    err << "layer \"" << get_name() << "\" has an "
        << "invalid number of input and output tensors "
        << "(found " << num_parents << " parent layers, "
        << num_children << " child layers, "
        << m_inputs.size() << " input tensors, "
        << m_outputs.size() << " output tensors, "
        << m_gradient_wrt_outputs.size() << " gradient w.r.t. output tensors, "
        << m_gradient_wrt_inputs.size() << " gradient w.r.t. input tensors)";
    LBANN_ERROR(err.str());
  }

  // Check that tensors are initialized
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_inputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized input tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_outputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized output tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (!m_gradient_wrt_outputs[i]) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. output tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    if (!m_gradient_wrt_inputs[i]) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. input tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::fp_setup_inputs(El::Int mini_batch_size) {
  if (get_num_parents() < 1) { return; }

  // Determine distributed matrix alignment
  const auto& alignment_dist = get_parent_layer().get_activations(*this).DistData();

  // Iterate through input tensors
  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_inputs(i)) continue;
#endif // LBANN_HAS_DISTCONV
    // Initialize input tensor
    const auto& parent = get_parent_layer(i);
    const auto& parent_output = parent.get_activations(*this);
    auto& input = *m_inputs[i];
    input.Empty(false);
    input.AlignWith(alignment_dist);
    if (parent_output.DistData() == input.DistData()) {
      El::LockedView(input, dynamic_cast<const AbsDistMatrixType&>(parent_output));
    } else {
      bool async_copy = false;
#if defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
      // Asynchronously copy CPU data to GPU data if they are otherwise aligned
      if (parent_output.GetLocalDevice() == El::Device::CPU
          && input.GetLocalDevice() == El::Device::GPU) {
        auto parent_dist_data = parent_output.DistData();
        parent_dist_data.device = El::Device::GPU;
        async_copy = parent_dist_data == input.DistData();
      }
#endif // defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
      if (async_copy) {
        El::CopyAsync(parent_output, input);
      } else {
        El::Copy(parent_output, input);
      }
    }

    // Check input matrix dimensions
    const auto& height = get_input_size(i);
    const auto& width = mini_batch_size;
    if (input.Height() != height || input.Width() != width) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "expected an input tensor stored in a "
          << height << " x " << width << " matrix "
          << "from layer \"" << parent.get_name() << "\", but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }

}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::fp_setup_outputs(El::Int mini_batch_size) {
  if (get_num_children() < 1) { return; }

  // Determine distributed matrix alignment
  const bool align_outputs = get_num_parents() > 0;
  const auto& alignment_dist = (align_outputs ?
                                get_prev_activations().DistData() :
                                get_activations().DistData());

  // Initialize output tensors
  for (int i = 0; i < get_num_children(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_outputs(i)) continue;
#endif // LBANN_HAS_DISTCONV
    auto& output = get_activations(i);
    output.Empty(false);
    if (align_outputs) { output.AlignWith(alignment_dist); }
    output.Resize(get_output_size(i), mini_batch_size);
  }

}

// Implementation details for back-propagation.
namespace {

// There's some strange logic for whether to do this copy
// asynchronously or not -- encapsulate it in this little function.
template <typename TDT>
void do_tensor_copy(const BaseDistMat& src,
                    El::AbstractDistMatrix<TDT>& tgt) {
  bool copy_async = false;
#if defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
  auto src_dist_data = src.DistData();
  auto tgt_dist_data = tgt.DistData();
  // Asynchronously copy CPU data to GPU data if they are otherwise aligned
  if ((src.dist_data.device == El::Device::CPU)
      && (tgt_dist_data.device == El::Device::GPU)) {
    src_dist_data.device = El::Device::GPU;
    copy_async = (src_dist_data == tgt_dist_data);
  }
#endif // defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
  if (copy_async) {
    El::CopyAsync(src, tgt);
  }
  else {
    El::Copy(src, tgt);
  }
}

// This was just cluttering up things.
void assert_tensor_size(const BaseDistMat& mat,
                        El::Int expected_height, El::Int expected_width,
                        std::string const& this_layer_name,
                        std::string const& child_layer_name)
{
  if ((mat.Height() != expected_height) || (mat.Width() != expected_width)) {
    LBANN_ERROR(
      "layer \"", this_layer_name, "\" expected a tensor stored in a ",
      expected_height, " x ", expected_width, " matrix from layer "
      "\"", child_layer_name, "\", but got a ",
      mat.Height(), " x ", mat.Width(), " matrix.");
  }
}

}// namespace <anon>

template <typename TensorDataType>
void data_type_layer<TensorDataType>::view_or_copy_prev_error_signal_(
  const Layer& child, const BaseDistMat& signal)
{
  auto layer_idx = find_child_layer_index(child);
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_gradient_wrt_outputs(layer_idx)) return;
#endif // LBANN_HAS_DISTCONV

  // Check the signal size
  assert_tensor_size(
    signal, get_output_size(layer_idx), m_outputs[layer_idx]->Width(),
    m_name, child.get_name());

  // If the distributions are compatible, we can just view
  // things. Otherwise, deep-copy the data.
  auto& prev_error_sig = *m_gradient_wrt_outputs[layer_idx];
  if (signal.DistData() == prev_error_sig.DistData()) {
    El::LockedView(prev_error_sig,
                   dynamic_cast<const AbsDistMatrixType&>(signal));
  }
  else {
    do_tensor_copy(signal, prev_error_sig);
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::move_or_copy_prev_error_signal_(
  const Layer& child, std::unique_ptr<BaseDistMat> signal_in)
{
    auto layer_idx = find_child_layer_index(child);
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_gradient_wrt_outputs(layer_idx)) return;
#endif // LBANN_HAS_DISTCONV

  // Check the signal size
  auto& signal = *signal_in;
  assert_tensor_size(
    signal, get_output_size(layer_idx), m_outputs[layer_idx]->Width(),
    m_name, child.get_name());

  // If the distribution is OK, then we can just swap data
  // around. Otherwise, deep copy into correct distribution.
  El::DistData expected_distdata = m_outputs[layer_idx]->DistData();
  if (signal.DistData() == expected_distdata) {
    if (auto sig_ptr = dynamic_cast<AbsDistMatrixType*>(signal_in.get())) {
      signal_in.release();
      m_gradient_wrt_outputs[layer_idx].reset(sig_ptr);
    }
    else {
      LBANN_ERROR("Logic error: DistData objects compare equal "
                  "but matrices have different dynamic types.");
    }
  }
  else // Deep copy
  {
    if (!m_gradient_wrt_outputs[layer_idx]) {
      m_gradient_wrt_outputs[layer_idx] =
        MakeMatBuilder<TensorDataType>(
          this->get_data_layout(),
          this->get_device_allocation())->MakeEmpty(*expected_distdata.grid, 0);
    }

    do_tensor_copy(signal, *m_gradient_wrt_outputs[layer_idx]);
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::deep_copy_prev_error_signal_(
  const Layer& child, const BaseDistMat& signal)
{
  auto layer_idx = find_child_layer_index(child);
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_gradient_wrt_outputs(layer_idx)) return;
#endif // LBANN_HAS_DISTCONV

  // Check the signal size
  assert_tensor_size(
    signal, get_output_size(layer_idx), m_outputs[layer_idx]->Width(),
    m_name, child.get_name());

  // If the distributions are compatible, we can just view
  // things. Otherwise, deep-copy the data.
  auto& prev_error_sig = *m_gradient_wrt_outputs[layer_idx];
  do_tensor_copy(signal, prev_error_sig);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::clear_prev_error_signals_() {
  if (!m_persistent_error_signals) {
    for (auto& es : m_gradient_wrt_outputs)
      es->Empty(true);
  }
}

void attempt_view_error_signal(
  Layer& parent, const Layer& child, const BaseDistMat& signal)
{
  parent.view_or_copy_prev_error_signal_(child, signal);
}

void attempt_move_error_signal(
  Layer& parent, const Layer& child, std::unique_ptr<BaseDistMat> signal)
{
  parent.move_or_copy_prev_error_signal_(child, std::move(signal));
}

void deep_copy_error_signal(
    Layer& parent, const Layer& child, const BaseDistMat& signal)
{
  parent.deep_copy_prev_error_signal_(child, signal);
}

// If I have persistent error signals, both my "previous error
// signals" and my new error signals will be persistent. So my parents
// can simply setup views into my error signals, if layout, alignment,
// etc is OK.
template <typename TensorDataType>
void data_type_layer<TensorDataType>::propagate_error_signals_to_parents_() {
  for (int i=0; i<get_num_parents(); ++i) {
    auto& parent = const_cast<Layer&>(get_parent_layer(i));

    // If my error signals persist, my parent can always view them,
    // assuming the distdata is right. Otherwise, my views and my data
    // will be released. Views must be copied and owned data can
    // either be copied or swapped out.
    auto& error_signal = *m_gradient_wrt_inputs[i];
    if (m_persistent_error_signals)
      attempt_view_error_signal(parent, *this, error_signal);
    else if (error_signal.Viewing())
      deep_copy_error_signal(parent, *this, error_signal);
    else
      attempt_move_error_signal(parent, *this,
                                std::move(m_gradient_wrt_inputs[i]));
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::allocate_new_gradients_() {
  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_gradient_wrt_inputs(i)) continue;
#endif // LBANN_HAS_DISTCONV
    if (!m_gradient_wrt_inputs[i]) {
      m_gradient_wrt_inputs[i] =
        MakeMatBuilder<TensorDataType>(
          this->get_data_layout(),
          this->get_device_allocation())->MakeEmpty(
            m_inputs[i]->Grid(), 0);
    }
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::bp_setup_gradient_wrt_inputs(
  El::Int mini_batch_size)
{
  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_gradient_wrt_inputs(i)) continue;
#endif // LBANN_HAS_DISTCONV
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
    gradient_wrt_input.Resize(get_input_size(i), mini_batch_size);
  }
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_distconv_adapter(const DataReaderMetaData& dr_metadata) {
  this->get_distconv_adapter_ptr() = make_unique<data_type_distconv_adapter<TensorDataType>>(*this);
}

template <typename TensorDataType>
data_type_distconv_adapter<TensorDataType>& data_type_layer<TensorDataType>::get_distconv_adapter() {
  return const_cast<data_type_distconv_adapter<TensorDataType>&>(
      static_cast<const data_type_layer<TensorDataType>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType>
const data_type_distconv_adapter<TensorDataType>& data_type_layer<TensorDataType>::get_distconv_adapter() const {
  return dynamic_cast<const data_type_distconv_adapter<TensorDataType>&>(*get_distconv_adapter_ptr());
}
#endif // LBANN_HAS_DISTCONV

#define PROTO(T)                     \
  template class data_type_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
