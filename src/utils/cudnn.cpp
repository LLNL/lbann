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

#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_CUDNN
#include "lbann/utils/number_theory.hpp"

#include "El.hpp"
#include <iostream>
#include <map>
#include <tuple>
#include <unordered_map>

#ifdef LBANN_HAS_CUDNN

namespace lbann {
namespace dnn_lib {

using namespace cudnn;

////////////////////////////////////////////////////////////
// Global cuDNN objects
////////////////////////////////////////////////////////////

namespace {

/** Wrapper for cuDNN handle. */
struct handle_wrapper
{
  cudnnHandle_t handle;
  handle_wrapper() : handle(nullptr)
  {
    CHECK_CUDA(cudaSetDevice(hydrogen::gpu::DefaultDevice()));
    if (handle == nullptr) {
      CHECK_CUDNN(cudnnCreate(&handle));
    }
    if (handle == nullptr) {
      LBANN_ERROR("failed to create cuDNN handle");
    }
    CHECK_CUDNN(cudnnSetStream(handle, hydrogen::cuda::GetDefaultStream()));
  }
  handle_wrapper(const handle_wrapper&) = delete;
  handle_wrapper& operator=(const handle_wrapper&) = delete;
  ~handle_wrapper()
  {
    if (handle != nullptr) {
      cudnnDestroy(handle);
    }
  }
};

/** Global instance of cuDNN handle. */
std::unique_ptr<handle_wrapper> handle_instance;

} // namespace

void initialize() { handle_instance.reset(new handle_wrapper()); }

void destroy() { handle_instance.reset(); }

cudnnHandle_t& get_handle()
{
  if (!handle_instance) {
    initialize();
  }
  CHECK_CUDA(cudaSetDevice(hydrogen::gpu::DefaultDevice()));
  CHECK_CUDNN(cudnnSetStream(handle_instance->handle,
                             hydrogen::cuda::GetDefaultStream()));
  return handle_instance->handle;
}

////////////////////////////////////////////////////////////
// Helper functions for cuDNN types
////////////////////////////////////////////////////////////

template <typename TensorDataType>
cudnnDataType_t get_data_type()
{
  LBANN_ERROR("invalid data type for cuDNN");
  return CUDNN_DATA_FLOAT;
}

#ifdef LBANN_HAS_GPU_FP16
template <>
cudnnDataType_t get_data_type<fp16>()
{
  return CUDNN_DATA_HALF;
}
#endif // LBANN_HAS_GPU_FP16
template <>
cudnnDataType_t get_data_type<float>()
{
  return CUDNN_DATA_FLOAT;
}
template <>
cudnnDataType_t get_data_type<double>()
{
  return CUDNN_DATA_DOUBLE;
}

////////////////////////////////////////////////////////////
// Wrapper classes for cuDNN types
////////////////////////////////////////////////////////////

// -----------------------------
// TensorDescriptor
// -----------------------------

TensorDescriptor::TensorDescriptor(cudnnTensorDescriptor_t desc) : desc_{desc}
{}

TensorDescriptor::~TensorDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    cudnnDestroyTensorDescriptor(desc_);
  }
}

TensorDescriptor::TensorDescriptor(const TensorDescriptor& other)
{
  if (other.desc_) {
    cudnnDataType_t data_type;
    int num_dims;
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(other.desc_,
                                           0, // nbDimsRequested
                                           &data_type,
                                           &num_dims,
                                           nullptr,   // dimA
                                           nullptr)); // strideA
    std::vector<int> dims(num_dims), strides(num_dims);
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(other.desc_,
                                           num_dims,
                                           &data_type,
                                           &num_dims,
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

void TensorDescriptor::reset(cudnnTensorDescriptor_t desc)
{
  if (desc_) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc_));
  }
  desc_ = desc;
}

cudnnTensorDescriptor_t TensorDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

cudnnTensorDescriptor_t TensorDescriptor::get() const noexcept { return desc_; }

TensorDescriptor::operator cudnnTensorDescriptor_t() const noexcept
{
  return get();
}

void TensorDescriptor::create()
{
  if (!desc_) {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc_));
  }
}

void TensorDescriptor::set(cudnnDataType_t data_type,
                           std::vector<int> dims_in,
                           std::vector<int> strides_in)
{

  // Check that arguments are valid
  if (dims_in.empty()) {
    LBANN_ERROR("attempted to set cuDNN tensor descriptor with no dimensions");
  }
  if (!strides_in.empty() && dims_in.size() != strides_in.size()) {
    LBANN_ERROR("attempted to set cuDNN tensor descriptor ",
                "with mismatched dimensions (",
                dims_in.size(),
                ") ",
                "and strides (",
                strides_in.size(),
                ")");
  }

  std::vector<int> dims = std::move(dims_in), strides = std::move(strides_in);
  if (dims.size() < 3) {
    dims.resize(3, 1);
    if (strides.size() < 3)
      strides.resize(3, 1);
  }

  // Assume data is contiguous if no strides are provided
  if (strides.empty()) {
    strides.resize(dims.size(), 1);
    for (int i = strides.size() - 1; i > 0; --i) {
      strides[i - 1] = strides[i] * dims[i];
    }
  }

  // Set cuDNN object
  create();
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc_,
                                         data_type,
                                         dims.size(),
                                         dims.data(),
                                         strides.data()));
}

// -----------------------------
// FilterDescriptor
// -----------------------------

FilterDescriptor::FilterDescriptor(cudnnFilterDescriptor_t desc) : desc_{desc}
{}

FilterDescriptor::~FilterDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    cudnnDestroyFilterDescriptor(desc_);
  }
}

FilterDescriptor::FilterDescriptor(const FilterDescriptor& other)
{
  if (other.desc_) {
    int num_dims;
    cudnnDataType_t data_type;
    cudnnTensorFormat_t format;
    std::vector<int> dims(1);
    CHECK_CUDNN(cudnnGetFilterNdDescriptor(other.desc_,
                                           dims.size(),
                                           &data_type,
                                           &format,
                                           &num_dims,
                                           dims.data()));
    dims.resize(num_dims);
    CHECK_CUDNN(cudnnGetFilterNdDescriptor(other.desc_,
                                           dims.size(),
                                           &data_type,
                                           &format,
                                           &num_dims,
                                           dims.data()));
    set(data_type, format, dims);
  }
}

FilterDescriptor::FilterDescriptor(FilterDescriptor&& other)
  : desc_{other.desc_}
{
  other.desc_ = nullptr;
}

FilterDescriptor& FilterDescriptor::operator=(FilterDescriptor other)
{
  swap(other, *this);
  return *this;
}

void swap(FilterDescriptor& first, FilterDescriptor& second)
{
  std::swap(first.desc_, second.desc_);
}

void FilterDescriptor::reset(cudnnFilterDescriptor_t desc)
{
  if (desc_) {
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(desc_));
  }
  desc_ = desc;
}

cudnnFilterDescriptor_t FilterDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

cudnnFilterDescriptor_t FilterDescriptor::get() const noexcept { return desc_; }

FilterDescriptor::operator cudnnFilterDescriptor_t() const noexcept
{
  return get();
}

void FilterDescriptor::create()
{
  if (!desc_) {
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc_));
  }
}

void FilterDescriptor::set(cudnnDataType_t data_type,
                           cudnnTensorFormat_t format,
                           const std::vector<int>& dims)
{
  create();
  CHECK_CUDNN(cudnnSetFilterNdDescriptor(desc_,
                                         data_type,
                                         format,
                                         dims.size(),
                                         dims.data()));
}

// -----------------------------
// DropoutDescriptor
// -----------------------------

DropoutDescriptor::DropoutDescriptor(cudnnDropoutDescriptor_t desc)
  : desc_{desc}
{}

DropoutDescriptor::~DropoutDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    cudnnDestroyDropoutDescriptor(desc_);
  }
}

DropoutDescriptor::DropoutDescriptor(const DropoutDescriptor& other)
{
  if (other.desc_) {
    float dropout;
    void* states;
    size_t states_size;
    unsigned long long seed;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(get_handle(), &states_size));
    CHECK_CUDNN(cudnnGetDropoutDescriptor(other.desc_,
                                          get_handle(),
                                          &dropout,
                                          &states,
                                          &seed));
    set(dropout, states, states_size, seed);
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

void DropoutDescriptor::reset(cudnnDropoutDescriptor_t desc)
{
  if (desc_) {
    CHECK_CUDNN(cudnnDestroyDropoutDescriptor(desc_));
  }
  desc_ = desc;
}

cudnnDropoutDescriptor_t DropoutDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

cudnnDropoutDescriptor_t DropoutDescriptor::get() const noexcept
{
  return desc_;
}

DropoutDescriptor::operator cudnnDropoutDescriptor_t() const noexcept
{
  return get();
}

void DropoutDescriptor::create()
{
  if (!desc_) {
    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&desc_));
  }
}

void DropoutDescriptor::set(float dropout,
                            void* states,
                            size_t states_size,
                            unsigned long long seed,
                            bool /*use_mask*/, // these 3 unused for cuDNN
                            bool /*state_evo*/,
                            int /*rng_mode*/)
{
  create();
  CHECK_CUDNN(cudnnSetDropoutDescriptor(desc_,
                                        get_handle(),
                                        dropout,
                                        states,
                                        states_size,
                                        seed));
}

// -----------------------------
// RNNDescriptor
// -----------------------------

RNNDescriptor::RNNDescriptor(cudnnRNNDescriptor_t desc) : desc_{desc} {}

RNNDescriptor::~RNNDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    cudnnDestroyRNNDescriptor(desc_);
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

void RNNDescriptor::reset(cudnnRNNDescriptor_t desc)
{
  if (desc_) {
    CHECK_CUDNN(cudnnDestroyRNNDescriptor(desc_));
  }
  desc_ = desc;
}

cudnnRNNDescriptor_t RNNDescriptor::release() noexcept
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

cudnnRNNDescriptor_t RNNDescriptor::get() const noexcept { return desc_; }

RNNDescriptor::operator cudnnRNNDescriptor_t() const noexcept { return get(); }

void RNNDescriptor::create()
{
  if (!desc_) {
    CHECK_CUDNN(cudnnCreateRNNDescriptor(&desc_));
  }
}

void RNNDescriptor::set(cudnnRNNAlgo_t algorithm,
                        cudnnRNNMode_t cell_mode,
                        cudnnRNNBiasMode_t bias_mode,
                        cudnnDirectionMode_t direction_mode,
                        cudnnRNNInputMode_t input_mode,
                        cudnnDataType_t data_type,
                        cudnnDataType_t math_precision,
                        cudnnMathType_t math_type,
                        size_t input_size,
                        size_t hidden_size,
                        size_t proj_size,
                        size_t num_layers,
                        cudnnDropoutDescriptor_t dropout_desc,
                        uint32_t aux_flags)
{
  create();
#if CUDNN_VERSION < 8000
  LBANN_ERROR("cuDNN 8 or newer is required for RNN support ",
              "(detected ",
              CUDNN_MAJOR,
              ".",
              CUDNN_MINOR,
              ".",
              CUDNN_PATCHLEVEL,
              ")");
#else  // CUDNN_VERSION >= 8000
  CHECK_CUDNN(cudnnSetRNNDescriptor_v8(desc_,
                                       algorithm,
                                       cell_mode,
                                       bias_mode,
                                       direction_mode,
                                       input_mode,
                                       data_type,
                                       math_precision,
                                       math_type,
                                       input_size,
                                       hidden_size,
                                       proj_size,
                                       num_layers,
                                       dropout_desc,
                                       aux_flags));
#endif // CUDNN_VERSION >= 8000
}

// -----------------------------
// RNNDataDescriptor
// -----------------------------

RNNDataDescriptor::RNNDataDescriptor(cudnnRNNDataDescriptor_t desc)
  : desc_{desc}
{}

RNNDataDescriptor::~RNNDataDescriptor()
{
  if (desc_) {
    // Don't check status to avoid exceptions
    cudnnDestroyRNNDataDescriptor(desc_);
  }
}

RNNDataDescriptor::RNNDataDescriptor(RNNDataDescriptor&& other)
  : desc_{other.desc_}
{
  other.desc_ = nullptr;
}

RNNDataDescriptor& RNNDataDescriptor::operator=(RNNDataDescriptor other)
{
  swap(other, *this);
  return *this;
}

void swap(RNNDataDescriptor& first, RNNDataDescriptor& second)
{
  std::swap(first.desc_, second.desc_);
}

void RNNDataDescriptor::reset(cudnnRNNDataDescriptor_t desc)
{
  if (desc_) {
    CHECK_CUDNN(cudnnDestroyRNNDataDescriptor(desc_));
  }
  desc_ = desc;
}

cudnnRNNDataDescriptor_t RNNDataDescriptor::release()
{
  auto old_desc = desc_;
  desc_ = nullptr;
  return old_desc;
}

cudnnRNNDataDescriptor_t RNNDataDescriptor::get() const noexcept
{
  return desc_;
}

RNNDataDescriptor::operator cudnnRNNDataDescriptor_t() const noexcept
{
  return get();
}

void RNNDataDescriptor::create()
{
  if (!desc_) {
    CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&desc_));
  }
}

void RNNDataDescriptor::set(cudnnDataType_t data_type,
                            cudnnRNNDataLayout_t layout,
                            size_t max_seq_length,
                            size_t batch_size,
                            size_t vector_size,
                            const int seq_length_array[],
                            void* padding_fill)
{
  create();
  CHECK_CUDNN(cudnnSetRNNDataDescriptor(desc_,
                                        data_type,
                                        layout,
                                        max_seq_length,
                                        batch_size,
                                        vector_size,
                                        seq_length_array,
                                        padding_fill));
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

  cudnnMathType_t math_type;
  cudnnConvolutionMode_t mode;
  cudnnDataType_t data_type;
  int num_dims, num_groups;
  CHECK_CUDNN(cudnnGetConvolutionMathType(rhs.desc_, &math_type));
  CHECK_CUDNN(cudnnGetConvolutionGroupCount(rhs.desc_, &num_groups));
  // Get the mode, data type, and dims
  CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(rhs.desc_,
                                              0,
                                              &num_dims,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              &mode,
                                              &data_type));

  // Get the padding, strides, and dilations
  std::vector<int> pads(num_dims), strides(num_dims), dilations(num_dims);
  CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(rhs.desc_,
                                              num_dims,
                                              &num_dims,
                                              pads.data(),
                                              strides.data(),
                                              dilations.data(),
                                              &mode,
                                              &data_type));

  // Now set up this one
  this->set(pads, strides, dilations, data_type, mode);
  this->set_math_mode(math_type);
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
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(desc_));

  desc_ = desc;
}

void ConvolutionDescriptor::create()
{
  if (!desc_)
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&desc_));
}

void ConvolutionDescriptor::set(std::vector<int> const& pad,
                                std::vector<int> const& stride,
                                std::vector<int> const& dilation,
                                cudnnDataType_t data_type,
                                cudnnConvolutionMode_t mode)
{
  LBANN_ASSERT(pad.size() == stride.size());
  LBANN_ASSERT(pad.size() == dilation.size());
  set(pad.size(), pad.data(), stride.data(), dilation.data(), data_type, mode);
}

void ConvolutionDescriptor::set(size_t array_dim,
                                int const pad[],
                                int const stride[],
                                int const dilation[],
                                cudnnDataType_t data_type,
                                cudnnConvolutionMode_t mode)
{
  this->create();
  CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(desc_,
                                              array_dim,
                                              pad,
                                              stride,
                                              dilation,
                                              mode,
                                              data_type));
}

void ConvolutionDescriptor::set_math_mode(cudnnMathType_t math_type)
{
  CHECK_CUDNN(cudnnSetConvolutionMathType(desc_, math_type));
}

void ConvolutionDescriptor::set_group_count(int num_groups)
{
  CHECK_CUDNN(cudnnSetConvolutionGroupCount(desc_, num_groups));
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

  cudnnPoolingMode_t mode;
  cudnnNanPropagation_t nan_prop;
  int num_dims;
  CHECK_CUDNN(cudnnGetPoolingNdDescriptor(rhs.desc_,
                                          0,
                                          &mode,
                                          &nan_prop,
                                          &num_dims,
                                          nullptr,
                                          nullptr,
                                          nullptr));
  std::vector<int> window_dims(num_dims), padding(num_dims), strides(num_dims);
  CHECK_CUDNN(cudnnGetPoolingNdDescriptor(rhs.desc_,
                                          num_dims,
                                          &mode,
                                          &nan_prop,
                                          &num_dims,
                                          window_dims.data(),
                                          padding.data(),
                                          strides.data()));
  this->set(cudnn::from_cudnn(mode), nan_prop, window_dims, padding, strides);
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
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(desc_));
  desc_ = desc;
}

void PoolingDescriptor::create()
{
  if (!desc_)
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&desc_));
}

void PoolingDescriptor::set(pooling_mode mode,
                            cudnnNanPropagation_t nan_prop,
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
                            cudnnNanPropagation_t nan_prop,
                            int num_dims,
                            int const window_dims[],
                            int const padding[],
                            int const stride[])
{
  this->create();
  CHECK_CUDNN(cudnnSetPoolingNdDescriptor(desc_,
                                          cudnn::to_cudnn(mode),
                                          nan_prop,
                                          num_dims,
                                          window_dims,
                                          padding,
                                          stride));
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
  CHECK_CUDNN(cudnnGetLRNDescriptor(desc_, &n, &alpha, &beta, &k));
  this->set(n, alpha, beta, k);
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
    CHECK_CUDNN(cudnnDestroyLRNDescriptor(desc_));
  desc_ = desc;
}

void LRNDescriptor::create()
{
  if (!desc_)
    CHECK_CUDNN(cudnnCreateLRNDescriptor(&desc_));
}

void LRNDescriptor::set(unsigned n,
                        double alpha,
                        double beta,
                        double k,
                        cudnnLRNMode_t mode)
{
  LBANN_ASSERT(n >= CUDNN_LRN_MIN_N);
  LBANN_ASSERT(n <= CUDNN_LRN_MAX_N);
  LBANN_ASSERT(k >= CUDNN_LRN_MIN_K);
  LBANN_ASSERT(beta >= CUDNN_LRN_MIN_BETA);

  this->create();
  CHECK_CUDNN(cudnnSetLRNDescriptor(desc_, n, alpha, beta, k));
}

/** @brief Swap two LRN descriptors. */
void swap(LRNDescriptor& lhs, LRNDescriptor& rhs) { lhs.swap(rhs); }

////////////////////////////////////////////////////////////
// Base cuDNN tensor manager
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
// Data-parallel cuDNN tensor manager
////////////////////////////////////////////////////////////

template <typename TensorDataType>
data_parallel_layer_tensor_manager<TensorDataType>::
  data_parallel_layer_tensor_manager(const data_type_layer<TensorDataType>* l)
  : layer_tensor_manager<TensorDataType>(l)
{}

namespace {

/** Set a cuDNN tensor descriptor for a data-parallel data layout.
 */
template <typename TensorDataType>
void set_data_parallel_tensor_desc(
  TensorDescriptor& desc,
  std::vector<int> dims,
  const El::AbstractMatrix<TensorDataType>& local_data)
{
#ifdef LBANN_DEBUG
  if (local_data.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
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
// Entry-wise cuDNN tensor manager
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
    LBANN_ERROR("attempted to setup cuDNN tensor with non-GPU data");
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
// cuDNN algorithm selection
////////////////////////////////////////////////////////////

namespace {

// Non-deterministic algorithms.
std::vector<cudnnConvolutionFwdAlgo_t> nondet_fwd_algos = {};
std::vector<cudnnConvolutionBwdDataAlgo_t> nondet_bwd_data_algos = {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0};
std::vector<cudnnConvolutionBwdFilterAlgo_t> nondet_bwd_filter_algos = {
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3};

template <typename AlgoType, typename PerfType>
AlgoType find_best_heuristic_algorithm(
  const std::vector<PerfType>& perf_results,
  const std::vector<AlgoType>& nondeterministic_algos,
  bool deterministic,
  size_t max_ws_size)
{
  std::vector<AlgoType> algos;
  for (const auto& p : perf_results) {
    if (p.status != CUDNN_STATUS_SUCCESS) {
      continue;
    }
    if (deterministic && std::find(nondeterministic_algos.begin(),
                                   nondeterministic_algos.end(),
                                   p.algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    algos.push_back(p.algo);
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
    if (p.status != CUDNN_STATUS_SUCCESS) {
      // If an algorithm fails, we still add it in case the failure is
      // nondeterministic.
      time_map[p.algo] = std::numeric_limits<float>::max();
      continue;
    }
    if (deterministic && std::find(nondeterministic_algos.begin(),
                                   nondeterministic_algos.end(),
                                   p.algo) != nondeterministic_algos.end()) {
      continue;
    }
    if (p.memory > max_ws_size) {
      continue;
    }
    if (time_map.count(p.algo) == 0) {
      time_map[p.algo] = p.time;
    }
    else {
      time_map[p.algo] += p.time;
    }
  }
  if (time_map.empty()) {
    LBANN_ERROR("No valid convolution algorithms.");
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

cudnnConvolutionFwdAlgo_t
get_fwd_algo_heuristic(bool deterministic,
                       const TensorDescriptor& input_desc,
                       const FilterDescriptor& kernel_desc,
                       const ConvolutionDescriptor& conv_desc,
                       const TensorDescriptor& output_desc,
                       size_t ws_size)
{
  int num_algos;
  CHECK_CUDNN(
    cudnnGetConvolutionForwardAlgorithmMaxCount(get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(get_handle(),
                                                     input_desc,
                                                     kernel_desc,
                                                     conv_desc,
                                                     output_desc,
                                                     num_algos,
                                                     &num_tested_algos,
                                                     perf_results.data()));
  return find_best_heuristic_algorithm(perf_results,
                                       nondet_fwd_algos,
                                       deterministic,
                                       ws_size);
}

cudnnConvolutionBwdDataAlgo_t
get_bwd_data_algo_heuristic(bool deterministic,
                            const FilterDescriptor& kernel_desc,
                            const TensorDescriptor& prev_error_signal_desc,
                            const ConvolutionDescriptor& conv_desc,
                            const TensorDescriptor& error_signal_desc,
                            size_t ws_size)
{
  int num_algos;
  CHECK_CUDNN(
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_CUDNN(
    cudnnGetConvolutionBackwardDataAlgorithm_v7(get_handle(),
                                                kernel_desc,
                                                prev_error_signal_desc,
                                                conv_desc,
                                                error_signal_desc,
                                                num_algos,
                                                &num_tested_algos,
                                                perf_results.data()));
  return find_best_heuristic_algorithm(perf_results,
                                       nondet_bwd_data_algos,
                                       deterministic,
                                       ws_size);
}

cudnnConvolutionBwdFilterAlgo_t
get_bwd_filter_algo_heuristic(bool deterministic,
                              const TensorDescriptor& input_desc,
                              const TensorDescriptor& prev_error_signal_desc,
                              const ConvolutionDescriptor& conv_desc,
                              const FilterDescriptor& kernel_gradient_desc,
                              size_t ws_size)
{
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(get_handle(),
                                                                 &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  int num_tested_algos;
  CHECK_CUDNN(
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(get_handle(),
                                                  input_desc,
                                                  prev_error_signal_desc,
                                                  conv_desc,
                                                  kernel_gradient_desc,
                                                  num_algos,
                                                  &num_tested_algos,
                                                  perf_results.data()));
  return find_best_heuristic_algorithm(perf_results,
                                       nondet_bwd_filter_algos,
                                       deterministic,
                                       ws_size);
}

cudnnConvolutionFwdAlgo_t
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
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  CHECK_CUDNN(
    cudnnGetConvolutionForwardAlgorithmMaxCount(get_handle(), &num_algos));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithmEx(get_handle(),
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
                                                       ws_size));
    if (trial >= num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all,
                             nondet_fwd_algos,
                             deterministic,
                             ws_size);
}

cudnnConvolutionBwdDataAlgo_t
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
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  CHECK_CUDNN(
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(get_handle(), &num_algos));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    CHECK_CUDNN(
      cudnnFindConvolutionBackwardDataAlgorithmEx(get_handle(),
                                                  kernel_desc,
                                                  kernel,
                                                  prev_error_signal_desc,
                                                  prev_error_signal,
                                                  conv_desc,
                                                  error_signal_desc,
                                                  error_signal,
                                                  num_algos,
                                                  &num_tested_algos,
                                                  perf_results.data(),
                                                  ws,
                                                  ws_size));
    if (trial >= num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all,
                             nondet_bwd_data_algos,
                             deterministic,
                             ws_size);
}

cudnnConvolutionBwdFilterAlgo_t
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
  constexpr int num_trials = 3;
  constexpr int num_skip = 1;
  int num_algos;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(get_handle(),
                                                                 &num_algos));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_all;
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(num_algos);
  for (int trial = 0; trial < num_trials + num_skip; ++trial) {
    int num_tested_algos;
    CHECK_CUDNN(
      cudnnFindConvolutionBackwardFilterAlgorithmEx(get_handle(),
                                                    input_desc,
                                                    input,
                                                    prev_error_signal_desc,
                                                    prev_error_signal,
                                                    conv_desc,
                                                    kernel_gradient_desc,
                                                    kernel_gradient,
                                                    num_algos,
                                                    &num_tested_algos,
                                                    perf_results.data(),
                                                    ws,
                                                    ws_size));
    if (trial >= num_skip) {
      for (const auto& p : perf_results) {
        perf_results_all.push_back(p);
      }
    }
  }
  return find_best_algorithm(perf_results_all,
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
  cudnnConvolutionFwdAlgo_t a;
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
                               kernel_desc,
                               conv_desc,
                               output_desc,
                               ws_size);
  }
  return from_cudnn(a);
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
  cudnnConvolutionBwdDataAlgo_t a;
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
                                    prev_error_signal_desc,
                                    conv_desc,
                                    error_signal_desc,
                                    ws_size);
  }
  return from_cudnn(a);
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
  cudnnConvolutionBwdFilterAlgo_t a;
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
                                      prev_error_signal_desc,
                                      conv_desc,
                                      kernel_gradient_desc,
                                      ws_size);
  }
  return from_cudnn(a);
}

namespace {
cudnnMathType_t default_tensor_ops_mode = CUDNN_TENSOR_OP_MATH;
}

void default_to_tensor_ops() noexcept
{
  default_tensor_ops_mode = CUDNN_TENSOR_OP_MATH;
}

void disable_tensor_ops() noexcept
{
  default_tensor_ops_mode = CUDNN_FMA_MATH;
}

cudnnMathType_t get_default_convolution_math_type() noexcept
{
  return default_tensor_ops_mode;
}

using ProtoTensorOpEnumType = decltype(lbann_data::DEFAULT_TENSOR_OPS);
cudnnMathType_t convert_to_dnn_math_type(ProtoTensorOpEnumType mt)
{
  switch (mt) {
  case lbann_data::DEFAULT_TENSOR_OPS:
    return dnn_lib::get_default_convolution_math_type();
  case lbann_data::NO_TENSOR_OPS:
    return CUDNN_FMA_MATH;
  case lbann_data::USE_TENSOR_OPS:
    return CUDNN_TENSOR_OP_MATH;
  case lbann_data::USE_TENSOR_OPS_ALLOW_CONVERSION:
    return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  default:
    LBANN_ERROR("Bad math type value.");
  }
  return CUDNN_FMA_MATH;
}

ProtoTensorOpEnumType convert_to_proto_math_type(cudnnMathType_t mt)
{
  switch (mt) {
  case CUDNN_DEFAULT_MATH:
  case CUDNN_FMA_MATH:
    // Note: These two are technically different in that DEFAULT_MATH
    // allows TF32 conversion, but we basically never want to use that.
    return lbann_data::NO_TENSOR_OPS;
  case CUDNN_TENSOR_OP_MATH:
    return lbann_data::USE_TENSOR_OPS;
  case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
    return lbann_data::USE_TENSOR_OPS_ALLOW_CONVERSION;
  default:
    return lbann_data::DEFAULT_TENSOR_OPS;
  }
}

std::string get_math_type_description(dnnMathType_t mt) {
  switch (mt) {
  case CUDNN_DEFAULT_MATH:
    return "No Tensor Cores + TF32 conversion";
  case CUDNN_FMA_MATH:
    return "No tensor cores";
  case CUDNN_TENSOR_OP_MATH:
    return "Tensor cores supported";
  case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
    return "Tensor cores + datatype downconversion";
  default:
    return "Unknown math type";
  }
}

template <typename TensorDataType>
dnnDataType_t get_convolution_data_type() {
  LBANN_ERROR("Invalid data type for cuDNN");
}

#ifdef LBANN_HAS_GPU_FP16
// Half should use float for the convolution descriptor.
// This corresponds to the PSEUDO_HALF_CONFIG in cuDNN.
template <>
dnnDataType_t get_convolution_data_type<fp16>() {
  return get_data_type<float>();
}
#endif
// Use the same type otherwise.
template <>
dnnDataType_t get_convolution_data_type<float>() {
  return get_data_type<float>();
}
template <>
dnnDataType_t get_convolution_data_type<double>() {
  return get_data_type<double>();
}

#ifdef LBANN_HAS_HALF
// Explicitly force gcc 10.3.1 to add a global symbol definition
// rather than optimizing it to a local symbol definition.
template cudnnDataType_t get_data_type<half_float::half>();
template cudnnDataType_t get_convolution_data_type<half_float::half>();
#endif

#define PROTO(T)                                                               \
  template class layer_tensor_manager<T>;                                      \
  template class data_parallel_layer_tensor_manager<T>;                        \
  template class entrywise_layer_tensor_manager<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace dnn_lib
} // namespace lbann

#endif // LBANN_HAS_CUDNN
