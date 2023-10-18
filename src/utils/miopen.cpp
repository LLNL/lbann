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

#include "lbann_config.hpp"

#ifdef LBANN_HAS_MIOPEN
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_MIOPEN
#include "lbann/utils/number_theory.hpp"

#include "El.hpp"
#include <iostream>
#include <map>
#include <tuple>
#include <unordered_map>

#ifdef LBANN_HAS_MIOPEN

namespace lbann {
namespace dnn_lib {

using namespace miopen;

////////////////////////////////////////////////////////////
// Global MIOpen objects
////////////////////////////////////////////////////////////

namespace {

/** Wrapper for cuDNN handle. */
struct handle_wrapper
{
  miopenHandle_t handle;
  handle_wrapper() : handle(nullptr)
  {
    CHECK_ROCM(hipSetDevice(hydrogen::gpu::DefaultDevice()));
    if (handle == nullptr) {
      CHECK_MIOPEN(miopenCreate(&handle));
    }
    if (handle == nullptr) {
      LBANN_ERROR("failed to create MIOpen handle");
    }
    CHECK_MIOPEN(miopenSetStream(handle, hydrogen::rocm::GetDefaultStream()));
  }
  handle_wrapper(const handle_wrapper&) = delete;
  handle_wrapper& operator=(const handle_wrapper&) = delete;
  ~handle_wrapper()
  {
    if (handle != nullptr) {
      miopenDestroy(handle);
    }
  }
};

/** Global instance of MIOpen handle. */
std::unique_ptr<handle_wrapper> handle_instance;

} // namespace

void initialize() { handle_instance.reset(new handle_wrapper()); }

void destroy() { handle_instance.reset(); }

miopenHandle_t& get_handle()
{
  if (!handle_instance) {
    initialize();
  }
  CHECK_ROCM(hipSetDevice(hydrogen::gpu::DefaultDevice()));
  CHECK_MIOPEN(miopenSetStream(handle_instance->handle,
                               hydrogen::rocm::GetDefaultStream()));
  return handle_instance->handle;
}

////////////////////////////////////////////////////////////
// Helper functions for MIOpen types
////////////////////////////////////////////////////////////

template <typename TensorDataType>
miopenDataType_t get_data_type()
{
  LBANN_ERROR("invalid data type for MIOpen");
  return miopenFloat;
}

#ifdef LBANN_HAS_GPU_FP16
template <>
miopenDataType_t get_data_type<fp16>()
{
  return miopenHalf;
}
#endif // LBANN_HAS_GPU_FP16
template <>
miopenDataType_t get_data_type<float>()
{
  return miopenFloat;
}
template <>
miopenDataType_t get_data_type<double>()
{
  LBANN_WARNING("Double is not supported in MIOpen");
  return miopenFloat;
}

////////////////////////////////////////////////////////////
// Wrapper classes for MIOpen types
////////////////////////////////////////////////////////////

// -----------------------------
// TensorDescriptor
// -----------------------------

TensorDescriptor::TensorDescriptor(miopenTensorDescriptor_t desc) : desc_{desc}
{}

TensorDescriptor::~TensorDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    miopenDestroyTensorDescriptor(desc_);
  }
}

TensorDescriptor::TensorDescriptor(const TensorDescriptor& other)
{
  if (other.desc_) {
    miopenDataType_t data_type;
    int num_dims;
    CHECK_MIOPEN(miopenGetTensorDescriptorSize(other.desc_, &num_dims));
    std::vector<int> dims(num_dims), strides(num_dims);
    CHECK_MIOPEN(miopenGetTensorDescriptor(other.desc_,
                                           &data_type,
                                           dims.data(),
                                           strides.data()));
    set(data_type, dims, strides);
  }
}

TensorDescriptor::TensorDescriptor(TensorDescriptor&& other)
  : desc_{other.desc_}
{
  other.desc_ = nullptr;
}

TensorDescriptor& TensorDescriptor::operator=(TensorDescriptor other)
{
  swap(other, *this);
  return *this;
}

void swap(TensorDescriptor& first, TensorDescriptor& second)
{
  std::swap(first.desc_, second.desc_);
}

void TensorDescriptor::reset(miopenTensorDescriptor_t desc)
{
  if (desc_) {
    CHECK_MIOPEN(miopenDestroyTensorDescriptor(desc_));
  }
  desc_ = desc;
}

miopenTensorDescriptor_t TensorDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

miopenTensorDescriptor_t TensorDescriptor::get() const noexcept
{
  return desc_;
}

TensorDescriptor::operator miopenTensorDescriptor_t() const noexcept
{
  return get();
}

void TensorDescriptor::create()
{
  if (!desc_) {
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&desc_));
  }
}

void TensorDescriptor::set(miopenDataType_t data_type,
                           std::vector<int> dims_in,
                           std::vector<int> strides_in)
{

  // Check that arguments are valid
  if (dims_in.empty()) {
    LBANN_ERROR("attempted to set MIOpen tensor descriptor with no dimensions");
  }
  if (!strides_in.empty() && dims_in.size() != strides_in.size()) {
    LBANN_ERROR("attempted to set MIOpen tensor descriptor ",
                "with mismatched dimensions (",
                dims_in.size(),
                ") ",
                "and strides (",
                strides_in.size(),
                ")");
  }

  std::vector<int> dims = std::move(dims_in), strides = std::move(strides_in);
  if (dims.size() < 4) {
    switch (dims.size()) {
    case 2:
      dims = {dims[0], 1, dims[1], 1};
      strides = {};
      break;
    case 3:
      dims = {dims[0], 1, dims[1], dims[2]};
      strides = {};
      break;
    default:
      LBANN_ERROR("Dims of size 1. Don't know what to do.");
      break;
    }
  }

  // Assume data is contiguous if no strides are provided
  // Note (trb 12/29/2020): MIOpen only accepts contiguous strides.
  if (strides.empty()) {
    strides.resize(dims.size(), 1);
    for (int i = strides.size() - 1; i > 0; --i) {
      strides[i - 1] = strides[i] * dims[i];
    }
  }

  // Set MIOpen object
  create();
  CHECK_MIOPEN(miopenSetTensorDescriptor(desc_,
                                         data_type,
                                         dims.size(),
                                         dims.data(),
                                         strides.data()));
}

// -----------------------------
// DropoutDescriptor
// -----------------------------

DropoutDescriptor::DropoutDescriptor(miopenDropoutDescriptor_t desc)
  : desc_{desc}
{}

DropoutDescriptor::~DropoutDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    miopenDestroyDropoutDescriptor(desc_);
  }
}

DropoutDescriptor::DropoutDescriptor(const DropoutDescriptor& other)
{
  if (other.desc_) {
    float dropout;
    void* states;
    size_t states_size;
    unsigned long long seed;
    bool use_mask, state_evo;
    miopenRNGType_t rng_mode;
    CHECK_MIOPEN(miopenDropoutGetStatesSize(get_handle(), &states_size));
    CHECK_MIOPEN(miopenGetDropoutDescriptor(other.desc_,
                                            get_handle(),
                                            &dropout,
                                            &states,
                                            &seed,
                                            &use_mask,
                                            &state_evo,
                                            &rng_mode));
    set(dropout, states, states_size, seed, use_mask, state_evo, rng_mode);
  }
}

DropoutDescriptor::DropoutDescriptor(DropoutDescriptor&& other)
  : desc_{other.desc_}
{
  other.desc_ = nullptr;
}

DropoutDescriptor& DropoutDescriptor::operator=(DropoutDescriptor other)
{
  swap(other, *this);
  return *this;
}

void swap(DropoutDescriptor& first, DropoutDescriptor& second)
{
  std::swap(first.desc_, second.desc_);
}

void DropoutDescriptor::reset(miopenDropoutDescriptor_t desc)
{
  if (desc_) {
    CHECK_MIOPEN(miopenDestroyDropoutDescriptor(desc_));
  }
  desc_ = desc;
}

miopenDropoutDescriptor_t DropoutDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

miopenDropoutDescriptor_t DropoutDescriptor::get() const noexcept
{
  return desc_;
}

DropoutDescriptor::operator miopenDropoutDescriptor_t() const noexcept
{
  return get();
}

void DropoutDescriptor::create()
{
  if (!desc_) {
    CHECK_MIOPEN(miopenCreateDropoutDescriptor(&desc_));
  }
}

void DropoutDescriptor::set(float dropout,
                            void* states,
                            size_t states_size,
                            unsigned long long seed,
                            bool use_mask,
                            bool state_evo,
                            miopenRNGType_t rng_mode)
{
  create();
  CHECK_MIOPEN(miopenSetDropoutDescriptor(desc_,
                                          get_handle(),
                                          dropout,
                                          states,
                                          states_size,
                                          seed,
                                          use_mask,
                                          state_evo,
                                          rng_mode));
}

// -----------------------------
// RNNDescriptor
// -----------------------------

RNNDescriptor::RNNDescriptor(miopenRNNDescriptor_t desc) : desc_{desc} {}

RNNDescriptor::~RNNDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    miopenDestroyRNNDescriptor(desc_);
  }
}

RNNDescriptor::RNNDescriptor(RNNDescriptor&& other) : desc_{other.desc_}
{
  other.desc_ = nullptr;
}

RNNDescriptor& RNNDescriptor::operator=(RNNDescriptor other)
{
  swap(other, *this);
  return *this;
}

void swap(RNNDescriptor& first, RNNDescriptor& second)
{
  std::swap(first.desc_, second.desc_);
}

void RNNDescriptor::reset(miopenRNNDescriptor_t desc)
{
  if (desc_) {
    CHECK_MIOPEN(miopenDestroyRNNDescriptor(desc_));
  }
  desc_ = desc;
}

miopenRNNDescriptor_t RNNDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

miopenRNNDescriptor_t RNNDescriptor::get() const noexcept { return desc_; }

RNNDescriptor::operator miopenRNNDescriptor_t() const noexcept { return get(); }

void RNNDescriptor::create()
{
  if (!desc_) {
    CHECK_MIOPEN(miopenCreateRNNDescriptor(&desc_));
  }
}

void RNNDescriptor::set(miopenRNNAlgo_t algorithm,
                        miopenRNNMode_t cell_mode,
                        miopenRNNBiasMode_t bias_mode,
                        miopenRNNDirectionMode_t direction_mode,
                        miopenRNNInputMode_t input_mode,
                        miopenDataType_t data_type,
                        miopenDataType_t math_precision,
                        int /*math_type*/, // placeholder
                        size_t input_size,
                        size_t hidden_size,
                        size_t proj_size,
                        size_t num_layers,
                        miopenDropoutDescriptor_t dropout_desc,
                        uint32_t aux_flags)
{
  create();
  CHECK_MIOPEN(miopenSetRNNDescriptor_V2(desc_,
                                         hidden_size,
                                         num_layers,
                                         dropout_desc,
                                         input_mode,
                                         direction_mode,
                                         cell_mode,
                                         bias_mode,
                                         algorithm,
                                         data_type));
}

// -----------------------------
// ConvolutionDescriptor
// -----------------------------

ConvolutionDescriptor::ConvolutionDescriptor(DescriptorHandle_t desc)
  : desc_{desc}
{}

ConvolutionDescriptor::~ConvolutionDescriptor()
{
  try {
    this->reset();
  }
  catch (std::exception const& e) {
    std::cerr << "Caught exception in ~ConvolutionDescriptor(). Shutting down."
              << std::endl;
    std::terminate();
  }
}

ConvolutionDescriptor::ConvolutionDescriptor(const ConvolutionDescriptor& rhs)
{
  // short-circuit
  if (rhs.desc_ == nullptr) {
    desc_ = nullptr;
    return;
  }

  miopenConvolutionMode_t mode;
  miopenDataType_t data_type; // placeholder
  data_type = miopenFloat;    // silence warning about uninitialized usage.
  int num_dims;
  // TODO: how to get group count?
  int num_groups = 1;
  // CHECK_MIOPEN(cudnnGetConvolutionGroupCount(rhs.desc_, &num_groups));
  //  Get the mode, data type, and dims
  CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(rhs.desc_,
                                                0,
                                                &num_dims,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                &mode));

  // Get the padding, strides, and dilations
  std::vector<int> pads(num_dims), strides(num_dims), dilations(num_dims);
  CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(rhs.desc_,
                                                num_dims,
                                                &num_dims,
                                                pads.data(),
                                                strides.data(),
                                                dilations.data(),
                                                &mode));

  // Now set up this one
  this->set(pads, strides, dilations, data_type, mode);
  this->set_group_count(num_groups);
}

ConvolutionDescriptor::ConvolutionDescriptor(ConvolutionDescriptor&& rhs)
  : desc_{rhs.desc_}
{
  rhs.desc_ = nullptr;
}

ConvolutionDescriptor&
ConvolutionDescriptor::operator=(ConvolutionDescriptor rhs)
{
  this->swap(rhs);
  return *this;
}

auto ConvolutionDescriptor::release() noexcept -> DescriptorHandle_t
{
  auto tmp = desc_;
  desc_ = nullptr;
  return tmp;
}

auto ConvolutionDescriptor::get() const noexcept -> DescriptorHandle_t
{
  return desc_;
}

ConvolutionDescriptor::operator DescriptorHandle_t() const noexcept
{
  return desc_;
}

void ConvolutionDescriptor::swap(ConvolutionDescriptor& other)
{
  std::swap(desc_, other.desc_);
}

void ConvolutionDescriptor::reset(DescriptorHandle_t desc)
{
  if (desc_)
    CHECK_MIOPEN(miopenDestroyConvolutionDescriptor(desc_));

  desc_ = desc;
}

void ConvolutionDescriptor::create()
{
  if (!desc_)
    CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&desc_));
}

void ConvolutionDescriptor::set(std::vector<int> const& pad,
                                std::vector<int> const& stride,
                                std::vector<int> const& dilation,
                                miopenDataType_t data_type,
                                miopenConvolutionMode_t mode)
{
  LBANN_ASSERT(pad.size() == stride.size());
  LBANN_ASSERT(pad.size() == dilation.size());
  set(pad.size(), pad.data(), stride.data(), dilation.data(), data_type, mode);
}

void ConvolutionDescriptor::set(size_t array_dim,
                                int const pad[],
                                int const stride[],
                                int const dilation[],
                                miopenDataType_t /*data_type*/,
                                miopenConvolutionMode_t mode)
{
  this->create();
  int* local_pad = const_cast<int*>(pad);
  int* local_stride = const_cast<int*>(stride);
  int* local_dilation = const_cast<int*>(dilation);
  CHECK_MIOPEN(miopenInitConvolutionNdDescriptor(desc_,
                                                 array_dim,
                                                 local_pad,
                                                 local_stride,
                                                 local_dilation,
                                                 mode));
}

void ConvolutionDescriptor::set_math_mode(int /*math_type*/) {}

void ConvolutionDescriptor::set_group_count(int num_groups)
{
  CHECK_MIOPEN(miopenSetConvolutionGroupCount(desc_, num_groups));
}

void swap(ConvolutionDescriptor& lhs, ConvolutionDescriptor& rhs)
{
  lhs.swap(rhs);
}

// -----------------------------
// PoolingDescriptor
// -----------------------------

PoolingDescriptor::PoolingDescriptor(DescriptorHandle_t desc) : desc_{desc} {}

PoolingDescriptor::~PoolingDescriptor()
{
  try {
    this->reset();
  }
  catch (std::exception const& e) {
    std::cerr << "Caught exception in ~PoolingDescriptor(). Shutting down."
              << std::endl;
    std::terminate();
  }
}

PoolingDescriptor::PoolingDescriptor(const PoolingDescriptor& rhs)
{
  if (rhs.desc_ == nullptr) {
    desc_ = nullptr;
    return;
  }

  miopenPoolingMode_t mode;
  miopenNanPropagation_t nan_prop;     // placeholder
  nan_prop = MIOPEN_NOT_PROPAGATE_NAN; // silence warning
  int num_dims;
  CHECK_MIOPEN(miopenGetNdPoolingDescriptor(rhs.desc_,
                                            0,
                                            &mode,
                                            &num_dims,
                                            nullptr,
                                            nullptr,
                                            nullptr));
  std::vector<int> window_dims(num_dims), padding(num_dims), strides(num_dims);
  CHECK_MIOPEN(miopenGetNdPoolingDescriptor(rhs.desc_,
                                            num_dims,
                                            &mode,
                                            &num_dims,
                                            window_dims.data(),
                                            padding.data(),
                                            strides.data()));
  this->set(miopen::from_miopen(mode), nan_prop, window_dims, padding, strides);
}

PoolingDescriptor::PoolingDescriptor(PoolingDescriptor&& rhs) : desc_{rhs.desc_}
{
  rhs.desc_ = nullptr;
}

PoolingDescriptor& PoolingDescriptor::operator=(PoolingDescriptor rhs)
{
  this->swap(rhs);
  return *this;
}

auto PoolingDescriptor::release() noexcept -> DescriptorHandle_t
{
  auto tmp = desc_;
  desc_ = nullptr;
  return tmp;
}

auto PoolingDescriptor::get() const noexcept -> DescriptorHandle_t
{
  return desc_;
}

PoolingDescriptor::operator DescriptorHandle_t() const noexcept
{
  return desc_;
}

void PoolingDescriptor::swap(PoolingDescriptor& other)
{
  std::swap(desc_, other.desc_);
}

void PoolingDescriptor::reset(DescriptorHandle_t desc)
{
  if (desc_)
    CHECK_MIOPEN(miopenDestroyPoolingDescriptor(desc_));
  desc_ = desc;
}

void PoolingDescriptor::create()
{
  if (!desc_)
    CHECK_MIOPEN(miopenCreatePoolingDescriptor(&desc_));
}

void PoolingDescriptor::set(pooling_mode mode,
                            miopenNanPropagation_t nan_prop, // placeholder
                            std::vector<int> const& window_dims,
                            std::vector<int> const& padding,
                            std::vector<int> const& stride)
{
  LBANN_ASSERT(window_dims.size() == padding.size());
  LBANN_ASSERT(window_dims.size() == stride.size());
  this->set(mode,
            nan_prop,
            window_dims.size(),
            window_dims.data(),
            padding.data(),
            stride.data());
}

void PoolingDescriptor::set(pooling_mode mode,
                            miopenNanPropagation_t nan_prop, // placeholder
                            int num_dims,
                            int const window_dims[],
                            int const padding[],
                            int const stride[])
{
  this->create();
  int* local_window_dims = const_cast<int*>(window_dims);
  int* local_padding = const_cast<int*>(padding);
  int* local_stride = const_cast<int*>(stride);
  CHECK_MIOPEN(miopenSetNdPoolingDescriptor(desc_,
                                            miopen::to_miopen(mode),
                                            num_dims,
                                            local_window_dims,
                                            local_padding,
                                            local_stride));
}

void swap(PoolingDescriptor& lhs, PoolingDescriptor& rhs) { lhs.swap(rhs); }

// -----------------------------
// LRNDescriptor
// -----------------------------

LRNDescriptor::LRNDescriptor(DescriptorHandle_t desc) : desc_{desc} {}

LRNDescriptor::~LRNDescriptor()
{
  try {
    this->reset();
  }
  catch (std::exception const& e) {
    std::cerr << "Caught exception in ~LRNDescriptor(). Shutting down."
              << std::endl;
    std::terminate();
  }
}

LRNDescriptor::LRNDescriptor(const LRNDescriptor& rhs)
{
  if (rhs.desc_ == nullptr) {
    desc_ = nullptr;
    return;
  }

  unsigned n;
  double alpha, beta, k;
  miopenLRNMode_t mode;
  CHECK_MIOPEN(miopenGetLRNDescriptor(desc_, &mode, &n, &alpha, &beta, &k));
  this->set(n, alpha, beta, k, mode);
}

LRNDescriptor::LRNDescriptor(LRNDescriptor&& rhs) : desc_{rhs.desc_}
{
  rhs.desc_ = nullptr;
}

LRNDescriptor& LRNDescriptor::operator=(LRNDescriptor rhs)
{
  this->swap(rhs);
  return *this;
}

auto LRNDescriptor::release() noexcept -> DescriptorHandle_t
{
  auto tmp = desc_;
  desc_ = nullptr;
  return tmp;
}

auto LRNDescriptor::get() const noexcept -> DescriptorHandle_t { return desc_; }

LRNDescriptor::operator DescriptorHandle_t() const noexcept { return desc_; }

void LRNDescriptor::swap(LRNDescriptor& other)
{
  std::swap(desc_, other.desc_);
}

void LRNDescriptor::reset(DescriptorHandle_t desc)
{
  if (desc_)
    CHECK_MIOPEN(miopenDestroyLRNDescriptor(desc_));
  desc_ = desc;
}

void LRNDescriptor::create()
{
  if (!desc_)
    CHECK_MIOPEN(miopenCreateLRNDescriptor(&desc_));
}

void LRNDescriptor::set(unsigned n,
                        double alpha,
                        double beta,
                        double k,
                        miopenLRNMode_t mode)
{
  // TODO: verify equivalent to CUDNN version
  LBANN_ASSERT(n >= 1);
  LBANN_ASSERT(n <= 16);
  LBANN_ASSERT(k >= 0.00001);
  LBANN_ASSERT(beta >= 0.01);

  this->create();
  CHECK_MIOPEN(miopenSetLRNDescriptor(desc_, mode, n, alpha, beta, k));
}

/** @brief Swap two LRN descriptors. */
void swap(LRNDescriptor& lhs, LRNDescriptor& rhs) { lhs.swap(rhs); }

////////////////////////////////////////////////////////////
// Base MIOpen tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
layer_tensor_manager<TensorDataType>::layer_tensor_manager(
  const data_type_layer<TensorDataType>* l)
  : m_layer(nullptr)
{
  set_layer(l);
}

template <typename TensorDataType>
void layer_tensor_manager<TensorDataType>::set_layer(
  const data_type_layer<TensorDataType>* new_layer)
{
  m_layer = new_layer;
  set_num_parents(this->m_layer == nullptr ? 0 : m_layer->get_num_parents());
  set_num_children(this->m_layer == nullptr ? 0 : m_layer->get_num_children());
}

template <typename TensorDataType>
void layer_tensor_manager<TensorDataType>::set_num_parents(int num_parents)
{
#ifdef LBANN_DEBUG
  if (num_parents < 0) {
    LBANN_ERROR("negative number of parents");
  }
#endif // LBANN_DEBUG
  for (size_t i = num_parents; i < m_prev_activations.size(); ++i) {
    if (m_prev_activations[i])
      m_prev_activations[i].reset();
  }
  for (size_t i = num_parents; i < m_error_signals.size(); ++i) {
    if (m_error_signals[i])
      m_error_signals[i].reset();
  }
  m_prev_activations.resize(num_parents);
  m_error_signals.resize(num_parents);
}

template <typename TensorDataType>
void layer_tensor_manager<TensorDataType>::set_num_children(int num_children)
{
#ifdef LBANN_DEBUG
  if (num_children < 0) {
    LBANN_ERROR("negative number of children");
  }
#endif // LBANN_DEBUG
  for (size_t i = num_children; i < m_activations.size(); ++i) {
    if (m_activations[i])
      m_activations[i].reset();
  }
  for (size_t i = num_children; i < m_prev_error_signals.size(); ++i) {
    if (m_prev_error_signals[i])
      m_prev_error_signals[i].reset();
  }
  m_activations.resize(num_children);
  m_prev_error_signals.resize(num_children);
}

////////////////////////////////////////////////////////////
// Data-parallel MIOpen tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
data_parallel_layer_tensor_manager<TensorDataType>::
  data_parallel_layer_tensor_manager(const data_type_layer<TensorDataType>* l)
  : layer_tensor_manager<TensorDataType>(l)
{}

namespace {

/** Set a MIOpen tensor descriptor for a data-parallel data layout.
 */
template <typename TensorDataType>
void set_data_parallel_tensor_desc(
  TensorDescriptor& desc,
  std::vector<int> dims,
  const El::AbstractMatrix<TensorDataType>& local_data)
{
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup MIOpen tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  if (local_data.Height() > 0 && local_data.Width() > 0) {
    std::vector<int> strides(dims.size(), 1);
    for (int i = strides.size() - 1; i > 0; --i) {
      strides[i - 1] = strides[i] * dims[i];
    }
    dims.insert(dims.begin(), local_data.Width());
    strides.insert(strides.begin(), local_data.LDim());
    desc.set(get_data_type<TensorDataType>(), dims, strides);
  }
}

} // namespace

template <typename TensorDataType>
TensorDescriptor&
data_parallel_layer_tensor_manager<TensorDataType>::get_prev_activations(
  int parent_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data =
    this->m_layer->get_local_prev_activations(parent_index);
  const auto& dims = this->m_layer->get_input_dims(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_prev_activations[parent_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
TensorDescriptor&
data_parallel_layer_tensor_manager<TensorDataType>::get_activations(
  int child_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_activations(child_index);
  const auto& dims = this->m_layer->get_output_dims(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_activations[child_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
TensorDescriptor&
data_parallel_layer_tensor_manager<TensorDataType>::get_prev_error_signals(
  int child_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data =
    this->m_layer->get_local_prev_error_signals(child_index);
  const auto& dims = this->m_layer->get_output_dims(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_prev_error_signals[child_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

template <typename TensorDataType>
TensorDescriptor&
data_parallel_layer_tensor_manager<TensorDataType>::get_error_signals(
  int parent_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_error_signals(parent_index);
  const auto& dims = this->m_layer->get_input_dims(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_error_signals[parent_index];
  set_data_parallel_tensor_desc<TensorDataType>(desc, dims, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// Entry-wise MIOpen tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
entrywise_layer_tensor_manager<TensorDataType>::entrywise_layer_tensor_manager(
  const data_type_layer<TensorDataType>* l)
  : layer_tensor_manager<TensorDataType>(l)
{}

namespace {

/** Set a cuDNN tensor descriptor for an entrywise tensor operation.
 *  Given local data in a (height x width) matrix, the tensor is
 *  initialized with dimensions (width, a, b, c), where
 *  a*b*c=height. This is because cuDNN is optimized for 4D tensors
 *  and gets poor performance with 1D tensors and 2D tensors.
 */
template <typename TensorDataType>
void set_entrywise_tensor_desc(
  TensorDescriptor& desc,
  const El::AbstractMatrix<TensorDataType>& local_data)
{
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup MIOpen tensor with non-GPU data");
  }
#endif // LBANN_DEBUG
  const int height = local_data.Height();
  const int width = local_data.Width();
  const int ldim = local_data.LDim();
  if (height > 0 && width > 0) {

    // Factorize height into three factors
    // Note: factorization is memoized
    static std::unordered_map<int, std::vector<int>> cache;
    auto& factors = cache[height];
    if (factors.empty()) {
      factors = number_theory::balanced_factors(height, 3);
    }

    // Set cuDNN tensor descriptor with 4D tensor
    desc.set(get_data_type<TensorDataType>(),
             {width, factors[2], factors[1], factors[0]},
             {ldim, factors[1] * factors[0], factors[0], 1});
  }
}

} // namespace

template <typename TensorDataType>
TensorDescriptor&
entrywise_layer_tensor_manager<TensorDataType>::get_prev_activations(
  int parent_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data =
    this->m_layer->get_local_prev_activations(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_prev_activations[parent_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

template <typename TensorDataType>
TensorDescriptor&
entrywise_layer_tensor_manager<TensorDataType>::get_activations(int child_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_activations(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_activations[child_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

template <typename TensorDataType>
TensorDescriptor&
entrywise_layer_tensor_manager<TensorDataType>::get_prev_error_signals(
  int child_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data =
    this->m_layer->get_local_prev_error_signals(child_index);
  this->set_num_children(this->m_layer->get_num_children());
  auto& desc = this->m_prev_error_signals[child_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

template <typename TensorDataType>
TensorDescriptor&
entrywise_layer_tensor_manager<TensorDataType>::get_error_signals(
  int parent_index)
{
  if (this->m_layer == nullptr) {
    LBANN_ERROR("tensor manager is not managing a layer");
  }
  const auto& local_data = this->m_layer->get_local_error_signals(parent_index);
  this->set_num_parents(this->m_layer->get_num_parents());
  auto& desc = this->m_error_signals[parent_index];
  set_entrywise_tensor_desc<TensorDataType>(desc, local_data);
  return desc;
}

////////////////////////////////////////////////////////////
// MIOpen algorithm selection
////////////////////////////////////////////////////////////

namespace {

// Non-deterministic algorithms.
std::vector<miopenConvFwdAlgorithm_t> nondet_fwd_algos = {};
std::vector<miopenConvBwdDataAlgorithm_t> nondet_bwd_data_algos = {
  // miopenConvolutionBwdDataAlgoGEMM
};
std::vector<miopenConvBwdWeightsAlgorithm_t> nondet_bwd_filter_algos = {
  // miopenConvolutionBwdWeightsAlgoGEMM
  // HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3
};

int ConvolutionFwdAlgoCount() { return 4; }
int ConvolutionBwdFilterAlgoCount() { return 2; }
int ConvolutionBwdDataAlgoCount() { return 2; }

template <typename AlgoType>
AlgoType get_miopen_conv_algo(const miopenConvAlgoPerf_t& perf_result)
{
  LBANN_ERROR("Convolution algorithm not supported by MIOpen");
}

template <>
miopenConvFwdAlgorithm_t get_miopen_conv_algo<miopenConvFwdAlgorithm_t>(
  const miopenConvAlgoPerf_t& perf_result)
{
  return perf_result.fwd_algo;
}

template <>
miopenConvBwdWeightsAlgorithm_t
get_miopen_conv_algo<miopenConvBwdWeightsAlgorithm_t>(
  const miopenConvAlgoPerf_t& perf_result)
{
  return perf_result.bwd_weights_algo;
}

template <>
miopenConvBwdDataAlgorithm_t get_miopen_conv_algo<miopenConvBwdDataAlgorithm_t>(
  const miopenConvAlgoPerf_t& perf_result)
{
  return perf_result.bwd_data_algo;
}

template <typename AlgoType, typename PerfType>
AlgoType find_best_heuristic_algorithm(
  const std::vector<PerfType>& perf_results,
  const std::vector<AlgoType>& nondeterministic_algos,
  bool deterministic,
  size_t max_ws_size)
{
  std::vector<AlgoType> algos;
  for (const auto& p : perf_results) {
    AlgoType p_algo = get_miopen_conv_algo<AlgoType>(p);
    if (deterministic && std::find(nondeterministic_algos.begin(),
                                   nondeterministic_algos.end(),
                                   p_algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    algos.push_back(p_algo);
  }
  if (algos.empty()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  return algos[0];
}

template <typename AlgoType, typename PerfType>
AlgoType
find_best_algorithm(const std::vector<PerfType>& perf_results,
                    const std::vector<AlgoType>& nondeterministic_algos,
                    bool deterministic,
                    size_t max_ws_size)
{
  std::map<AlgoType, float> time_map;
  for (const auto& p : perf_results) {
    AlgoType p_algo = get_miopen_conv_algo<AlgoType>(p);
    if (deterministic && std::find(nondeterministic_algos.begin(),
                                   nondeterministic_algos.end(),
                                   p_algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    if (time_map.count(p_algo) == 0) {
      time_map[p_algo] = p.time;
    }
    else {
      time_map[p_algo] += p.time;
    }
  }
  if (time_map.empty()) {
    if (deterministic) {
      LBANN_WARNING("No valid deterministic convolution algorithms, "
                    "trying again with deterministic=false");
      return find_best_algorithm(perf_results,
                                 nondeterministic_algos,
                                 false,
                                 max_ws_size);
    }
    else {
      LBANN_ERROR("No valid convolution algorithms.");
    }
  }
  AlgoType best_algo = time_map.begin()->first;
  float min_time = std::numeric_limits<float>::max();
  for (const auto& x : time_map) {
    AlgoType algo = x.first;
    float time = x.second;
    if (time < min_time) {
      min_time = time;
      best_algo = algo;
    }
  }
  if (min_time == std::numeric_limits<float>::max()) {
    LBANN_ERROR("No valid convolution algorithms.");
  }
  return best_algo;
}

miopenConvFwdAlgorithm_t
get_fwd_algo_heuristic(bool deterministic,
                       const TensorDescriptor& input_desc,
                       const void* input,
                       const FilterDescriptor& kernel_desc,
                       const void* kernel,
                       const ConvolutionDescriptor& conv_desc,
                       const TensorDescriptor& output_desc,
                       void* output,
                       size_t ws_size,
                       void* ws)
{
  int num_algos = ConvolutionFwdAlgoCount();
  std::vector<miopenConvAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(get_handle(),
                                                     input_desc,
                                                     input,
                                                     kernel_desc,
                                                     kernel,
                                                     conv_desc,
                                                     output_desc,
                                                     output,
                                                     num_algos,
                                                     &num_tested_algos,
                                                     perf_results.data(),
                                                     ws,
                                                     ws_size,
                                                     true));
  perf_results.resize(num_tested_algos);
  return find_best_heuristic_algorithm(perf_results,
                                       nondet_fwd_algos,
                                       deterministic,
                                       ws_size);
}

miopenConvBwdDataAlgorithm_t
get_bwd_data_algo_heuristic(bool deterministic,
                            const FilterDescriptor& kernel_desc,
                            const void* kernel,
                            const TensorDescriptor& prev_error_signal_desc,
                            const void* prev_error_signal,
                            const ConvolutionDescriptor& conv_desc,
                            const TensorDescriptor& error_signal_desc,
                            void* error_signal,
                            size_t ws_size,
                            void* ws)
{
  int num_algos = ConvolutionBwdDataAlgoCount();
  std::vector<miopenConvAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_MIOPEN(
    miopenFindConvolutionBackwardDataAlgorithm(get_handle(),
                                               prev_error_signal_desc,
                                               prev_error_signal,
                                               kernel_desc,
                                               kernel,
                                               conv_desc,
                                               error_signal_desc,
                                               error_signal,
                                               num_algos,
                                               &num_tested_algos,
                                               perf_results.data(),
                                               ws,
                                               ws_size,
                                               true));
  perf_results.resize(num_tested_algos);
  return find_best_heuristic_algorithm(perf_results,
                                       nondet_bwd_data_algos,
                                       deterministic,
                                       ws_size);
}

miopenConvBwdWeightsAlgorithm_t
get_bwd_filter_algo_heuristic(bool deterministic,
                              const TensorDescriptor& input_desc,
                              const void* input,
                              const TensorDescriptor& prev_error_signal_desc,
                              const void* prev_error_signal,
                              const ConvolutionDescriptor& conv_desc,
                              const FilterDescriptor& kernel_gradient_desc,
                              void* kernel_gradient,
                              size_t ws_size,
                              void* ws)
{
  int num_algos = ConvolutionBwdFilterAlgoCount();
  std::vector<miopenConvAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_MIOPEN(
    miopenFindConvolutionBackwardWeightsAlgorithm(get_handle(),
                                                  prev_error_signal_desc,
                                                  prev_error_signal,
                                                  input_desc,
                                                  input,
                                                  conv_desc,
                                                  kernel_gradient_desc,
                                                  kernel_gradient,
                                                  num_algos,
                                                  &num_tested_algos,
                                                  perf_results.data(),
                                                  ws,
                                                  ws_size,
                                                  true));
  perf_results.resize(num_tested_algos);
  return find_best_heuristic_algorithm(perf_results,
                                       nondet_bwd_filter_algos,
                                       deterministic,
                                       ws_size);
}

miopenConvFwdAlgorithm_t
get_fwd_algo_autotune(bool deterministic,
                      const TensorDescriptor& input_desc,
                      const void* input,
                      const FilterDescriptor& kernel_desc,
                      const void* kernel,
                      const ConvolutionDescriptor& conv_desc,
                      const TensorDescriptor& output_desc,
                      void* output,
                      size_t ws_size,
                      void* ws)
{
  int num_algos = ConvolutionFwdAlgoCount();
  std::vector<miopenConvAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(get_handle(),
                                                     input_desc,
                                                     input,
                                                     kernel_desc,
                                                     kernel,
                                                     conv_desc,
                                                     output_desc,
                                                     output,
                                                     num_algos,
                                                     &num_tested_algos,
                                                     perf_results.data(),
                                                     ws,
                                                     ws_size,
                                                     false));
  perf_results.resize(num_tested_algos);
  return find_best_algorithm(perf_results,
                             nondet_fwd_algos,
                             deterministic,
                             ws_size);
}

miopenConvBwdDataAlgorithm_t
get_bwd_data_algo_autotune(bool deterministic,
                           const FilterDescriptor& kernel_desc,
                           const void* kernel,
                           const TensorDescriptor& prev_error_signal_desc,
                           const void* prev_error_signal,
                           const ConvolutionDescriptor& conv_desc,
                           const TensorDescriptor& error_signal_desc,
                           void* error_signal,
                           size_t ws_size,
                           void* ws)
{
  int num_algos = ConvolutionBwdDataAlgoCount();
  std::vector<miopenConvAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_MIOPEN(
    miopenFindConvolutionBackwardDataAlgorithm(get_handle(),
                                               prev_error_signal_desc,
                                               prev_error_signal,
                                               kernel_desc,
                                               kernel,
                                               conv_desc,
                                               error_signal_desc,
                                               error_signal,
                                               num_algos,
                                               &num_tested_algos,
                                               perf_results.data(),
                                               ws,
                                               ws_size,
                                               false));
  perf_results.resize(num_tested_algos);
  return find_best_algorithm(perf_results,
                             nondet_bwd_data_algos,
                             deterministic,
                             ws_size);
}

miopenConvBwdWeightsAlgorithm_t
get_bwd_filter_algo_autotune(bool deterministic,
                             const TensorDescriptor& input_desc,
                             const void* input,
                             const TensorDescriptor& prev_error_signal_desc,
                             const void* prev_error_signal,
                             const ConvolutionDescriptor& conv_desc,
                             const FilterDescriptor& kernel_gradient_desc,
                             void* kernel_gradient,
                             size_t ws_size,
                             void* ws)
{
  int num_algos = ConvolutionBwdFilterAlgoCount();
  std::vector<miopenConvAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_MIOPEN(
    miopenFindConvolutionBackwardWeightsAlgorithm(get_handle(),
                                                  prev_error_signal_desc,
                                                  prev_error_signal,
                                                  input_desc,
                                                  input,
                                                  conv_desc,
                                                  kernel_gradient_desc,
                                                  kernel_gradient,
                                                  num_algos,
                                                  &num_tested_algos,
                                                  perf_results.data(),
                                                  ws,
                                                  ws_size,
                                                  false));
  perf_results.resize(num_tested_algos);
  return find_best_algorithm(perf_results,
                             nondet_bwd_filter_algos,
                             deterministic,
                             ws_size);
}

} // namespace

fwd_conv_alg get_fwd_algorithm(bool autotune,
                               bool deterministic,
                               const TensorDescriptor& input_desc,
                               const void* input,
                               const FilterDescriptor& kernel_desc,
                               const void* kernel,
                               const ConvolutionDescriptor& conv_desc,
                               const TensorDescriptor& output_desc,
                               void* output,
                               size_t ws_size,
                               void* ws)
{
  miopenConvFwdAlgorithm_t a;
  if (autotune) {
    a = get_fwd_algo_autotune(deterministic,
                              input_desc,
                              input,
                              kernel_desc,
                              kernel,
                              conv_desc,
                              output_desc,
                              output,
                              ws_size,
                              ws);
  }
  else {
    a = get_fwd_algo_heuristic(deterministic,
                               input_desc,
                               input,
                               kernel_desc,
                               kernel,
                               conv_desc,
                               output_desc,
                               output,
                               ws_size,
                               ws);
  }
  return from_miopen(a);
}

bwd_data_conv_alg
get_bwd_data_algorithm(bool autotune,
                       bool deterministic,
                       const FilterDescriptor& kernel_desc,
                       const void* kernel,
                       const TensorDescriptor& prev_error_signal_desc,
                       const void* prev_error_signal,
                       const ConvolutionDescriptor& conv_desc,
                       const TensorDescriptor& error_signal_desc,
                       void* error_signal,
                       size_t ws_size,
                       void* ws)
{
  miopenConvBwdDataAlgorithm_t a;
  if (autotune) {
    a = get_bwd_data_algo_autotune(deterministic,
                                   kernel_desc,
                                   kernel,
                                   prev_error_signal_desc,
                                   prev_error_signal,
                                   conv_desc,
                                   error_signal_desc,
                                   error_signal,
                                   ws_size,
                                   ws);
  }
  else {
    a = get_bwd_data_algo_heuristic(deterministic,
                                    kernel_desc,
                                    kernel,
                                    prev_error_signal_desc,
                                    prev_error_signal,
                                    conv_desc,
                                    error_signal_desc,
                                    error_signal,
                                    ws_size,
                                    ws);
  }
  return from_miopen(a);
}

bwd_filter_conv_alg
get_bwd_filter_algorithm(bool autotune,
                         bool deterministic,
                         const TensorDescriptor& input_desc,
                         const void* input,
                         const TensorDescriptor& prev_error_signal_desc,
                         const void* prev_error_signal,
                         const ConvolutionDescriptor& conv_desc,
                         const FilterDescriptor& kernel_gradient_desc,
                         void* kernel_gradient,
                         size_t ws_size,
                         void* ws)
{
  miopenConvBwdWeightsAlgorithm_t a;
  if (autotune) {
    a = get_bwd_filter_algo_autotune(deterministic,
                                     input_desc,
                                     input,
                                     prev_error_signal_desc,
                                     prev_error_signal,
                                     conv_desc,
                                     kernel_gradient_desc,
                                     kernel_gradient,
                                     ws_size,
                                     ws);
  }
  else {
    a = get_bwd_filter_algo_heuristic(deterministic,
                                      input_desc,
                                      input,
                                      prev_error_signal_desc,
                                      prev_error_signal,
                                      conv_desc,
                                      kernel_gradient_desc,
                                      kernel_gradient,
                                      ws_size,
                                      ws);
  }
  return from_miopen(a);
}

// Placeholder functions for mathtype
namespace {
int default_tensor_ops_mode = 0;
}

void default_to_tensor_ops() noexcept { default_tensor_ops_mode = 0; }

int get_default_convolution_math_type() noexcept { return 0; }

using ProtoTensorOpEnumType = decltype(lbann_data::DEFAULT_TENSOR_OPS);
int convert_to_dnn_math_type(ProtoTensorOpEnumType mt)
{
  // MIOpen only supports one math type, so this is just a placeholder function
  return 0;
}

ProtoTensorOpEnumType convert_to_proto_math_type(dnnMathType_t mt)
{
  return lbann_data::DEFAULT_TENSOR_OPS;
}

std::string get_math_type_description(dnnMathType_t mt) {
  return "MIOpen math";  // MIOpen does not have different math types.
}

// MIOpen does not use a datatype in its convolution descriptor but we
// mirror cuDNN here.
template <typename TensorDataType>
dnnDataType_t get_convolution_data_type() {
  LBANN_ERROR("Invalid data type for MIOpen");
}
#ifdef LBANN_HAS_GPU_FP16
template <>
dnnDataType_t get_convolution_data_type<fp16>() {
  return get_data_type<float>();
}
#endif
template <>
dnnDataType_t get_convolution_data_type<float>() {
  return get_data_type<float>();
}
template <>
dnnDataType_t get_convolution_data_type<double>() {
  LBANN_WARNING("MIOpen does not support double");
  return get_data_type<float>();
}

#define PROTO(T)                                                               \
  template class layer_tensor_manager<T>;                                      \
  template class data_parallel_layer_tensor_manager<T>;                        \
  template class entrywise_layer_tensor_manager<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace dnn_lib
} // namespace lbann

#endif // LBANN_HAS_MIOPEN
