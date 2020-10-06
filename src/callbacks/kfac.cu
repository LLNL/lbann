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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/kfac.hpp"


namespace lbann {
namespace callback {

namespace {

template <typename TensorDataType>
__global__ void kfac_add_to_diagonal_kernel(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType value,
    const TensorDataType value_bn_err,
    const bool is_bn) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height) {
    A[gid+gid*height] += (is_bn && gid >= height/2 ? value_bn_err : value);
  }
}

template <typename TensorDataType>
__global__ void kfac_fill_upper_tri_kernel(
    TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t row = gid%height, col = gid/height;
  if(row < height && col < height && row < col) {
    A[row+col*height] += A[col+row*height];
  }
}

template <typename TensorDataType>
__global__ void kfac_update_kronecker_average_kernel(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count, const double decay) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    Aave[gid] = Aave[gid]*decay + A[gid]*(1.0-decay);
  }
}

#ifdef LBANN_HAS_HALF
template <>
__global__ void kfac_update_kronecker_average_kernel<__half>(
    __half * __restrict__ Aave,
    const __half * __restrict__ A,
    const size_t count, const double decay) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < count) {
    Aave[gid] = (float) Aave[gid]*decay + (float) A[gid]*(1.0-decay);
  }
}
#endif // LBANN_HAS_HALF

template <typename TensorDataType>
__global__ void kfac_conv_transpose_kernel(
    const TensorDataType * __restrict__ A,
    TensorDataType * __restrict__ Acol,
    const size_t mini_batch_size, const size_t num_channels,
    const size_t spatial_prod, const size_t num_elems) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_elems) {
    const auto i_spatial = gid%spatial_prod;
    const auto i_c = (gid/spatial_prod)%num_channels;
    const auto i_n = (gid/spatial_prod/num_channels);
    Acol[i_c+i_spatial*num_channels+i_n*num_channels*spatial_prod] = A[gid];
  }
}

template <typename TensorDataType>
__global__ void kfac_compute_bn_factor_kernel(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ factor,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod,
    const size_t num_threads) { // = batch_size*num_channels
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < num_threads) {
    const size_t i_c = gid%num_channels;
    const size_t i_n = gid/num_channels;

    const auto scale = scales[i_c];
    const auto bias = biases[i_c];

    TensorDataType sum_ea = 0.0, sum_e = 0.0;
    // TODO: This loop would be slow in large (3D) CNNs
    for(size_t i_s = 0; i_s < spatial_prod; i_s++) {
      const auto i_act = i_s+gid*spatial_prod;
      const auto error = errors[i_act];
      const auto act = (activations[i_act]-bias)/scale;
      sum_ea += error * act;
      sum_e += error;
    }
    factor[i_c+i_n*num_channels*2] = sum_ea;
    factor[i_c+num_channels+i_n*num_channels*2] = sum_e;
  }
}

template <typename TensorDataType>
__global__ void kfac_identity_kernel(
    TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height*height) {
    const size_t row = gid%height;
    const size_t col = gid/height;
    A[gid] = (row == col ? 1.0 : 0.0);
  }
}

template <typename TensorDataType>
__global__ void kfac_pack_lower_tri_kernel(
    TensorDataType * __restrict__ L,
    const TensorDataType * __restrict__ A,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height*height) {
    const size_t row = gid%height;
    const size_t col = gid/height;
    if(row >= col) {
      L[row+(2*height-(col-1))*col/2-col] = A[gid];
    }
  }
}

template <typename TensorDataType>
__global__ void kfac_unpack_lower_tri_kernel(
    TensorDataType * __restrict__ A,
    const TensorDataType * __restrict__ L,
    const size_t height) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < height*height) {
    const size_t row = gid%height;
    const size_t col = gid/height;
    if(row >= col) {
      A[gid] = A[col+row*height] = L[row+(2*height-(col-1))*col/2-col];
    }
  }
}

} // namespace

template <typename TensorDataType>
void kfac::add_to_diagonal(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType damping,
    const TensorDataType damping_bn_err,
    const bool is_bn,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (height + block_size - 1) / block_size;
  kfac_add_to_diagonal_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      A, height, damping,
      damping_bn_err, is_bn);
}

template <typename TensorDataType>
void kfac::fill_upper_tri(
    TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  // TODO: Launch N^2/2 threads instead of N^2
  const size_t grid_size = (height*height + block_size - 1) / block_size;
  kfac_fill_upper_tri_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      A, height);
}

template <typename TensorDataType>
void kfac::update_kronecker_average(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count, const double decay,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  kfac_update_kronecker_average_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      Aave, A, count, decay);
}

template <typename TensorDataType>
void kfac::conv_transpose(
    const TensorDataType * __restrict__ activations,
    TensorDataType * __restrict__ act_columns,
    const size_t mini_batch_size, const size_t num_channels,
    const size_t spatial_prod,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_elems = mini_batch_size*num_channels*spatial_prod;
  const size_t grid_size = (num_elems + block_size - 1) / block_size;
  kfac_conv_transpose_kernel<TensorDataType><<<grid_size, block_size, 0, stream>>>(
      activations, act_columns, mini_batch_size, num_channels, spatial_prod,
      num_elems);
}

template <typename TensorDataType>
void kfac::compute_bn_factor(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ factor,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = batch_size * num_channels;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  kfac_compute_bn_factor_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          activations, errors,
          scales, biases,
          factor,
          batch_size, num_channels, spatial_prod,
          num_threads);
}

template <typename TensorDataType>
void kfac::identity(
    TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height*height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  kfac_identity_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          A, height);
}

template <typename TensorDataType>
void kfac::pack_lower_tri(
    TensorDataType * __restrict__ L,
    const TensorDataType * __restrict__ A,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height*height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  // TODO: Launch N^2/2 threads instead of N^2
  kfac_pack_lower_tri_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          L, A, height);
}

template <typename TensorDataType>
void kfac::unpack_lower_tri(
    TensorDataType * __restrict__ A,
    const TensorDataType * __restrict__ L,
    const size_t height,
    const cudaStream_t& stream) {
  constexpr size_t block_size = 256;
  const size_t num_threads = height*height;
  const size_t grid_size = (num_threads + block_size - 1) / block_size;
  // TODO: Launch N^2/2 threads instead of N^2
  kfac_unpack_lower_tri_kernel<TensorDataType>
      <<<grid_size, block_size, 0, stream>>>(
          A, L, height);
}

#define PROTO(T)                                        \
  template void kfac::add_to_diagonal<T>(               \
      T* __restrict__ A,                                \
      const size_t height,                              \
      const T value,                                    \
      const T value_bn_err,                             \
      const bool is_bn,                                 \
      const cudaStream_t& stream);                      \
  template void kfac::fill_upper_tri<T>(                \
      T * __restrict__ A,                               \
      const size_t height,                              \
      const cudaStream_t& stream);                      \
  template void kfac::update_kronecker_average<T>(      \
      T * __restrict__ Aave,                            \
      const T * __restrict__ A,                         \
      const size_t count, const double decay,           \
      const cudaStream_t& stream);                      \
  template void kfac::conv_transpose<T>(                \
      const T * __restrict__ activations,               \
      T * __restrict__ act_columns,                     \
      const size_t mini_batch_size,                     \
      const size_t num_channels,                        \
      const size_t spatial_prod,                        \
      const cudaStream_t& stream);                      \
  template void kfac::compute_bn_factor<T>(             \
      const T * __restrict__ activations,               \
      const T * __restrict__ errors,                    \
      const T * __restrict__ scales,                    \
      const T * __restrict__ biases,                    \
      T * __restrict__ factor,                          \
      const size_t batch_size,                          \
      const size_t num_channels,                        \
      const size_t spatial_prod,                        \
      const cudaStream_t& stream);                      \
  template void kfac::identity<T>(                      \
      T * __restrict__ A,                               \
      const size_t height,                              \
      const cudaStream_t& stream);                      \
  template void kfac::pack_lower_tri<T>(                \
      T * __restrict__ L,                               \
      const T * __restrict__ A,                         \
      const size_t height,                              \
      const cudaStream_t& stream);                      \
  template void kfac::unpack_lower_tri<T>(              \
      T * __restrict__ A,                               \
      const T * __restrict__ L,                         \
      const size_t height,                              \
      const cudaStream_t& stream)


#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

    } // namespace callback
} // namespace lbann
