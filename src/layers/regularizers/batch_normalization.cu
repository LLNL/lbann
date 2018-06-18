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

#include "math.h"
#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {

namespace {

// Atomic add functions
#if __CUDA_ARCH__ >= 530
__device__ inline __half atomic_add(__half* address, __half val) {
#if 0 // TODO: replace this once Nvidia implements atomicAdd for __half
  return atomicAdd(address, val);
#else
  unsigned int* address_as_uint = (unsigned int*) address;
  unsigned int old = *address_as_uint;
  __half* old_as_half = (__half*) &old;
  unsigned int assumed;
  unsigned int updated;
  __half* updated_as_half = (__half*) &updated;
  do {
    assumed = old;
    updated = old;
    *updated_as_half += value;
    old = atomicCAS(address_as_uint, assumed, updated);
  } while (assumed != old);
  return *old_as_half;
#endif // 0
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float atomic_add(float* address, float val) {
  return atomicAdd(address, val);
}
__device__ inline double atomic_add(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif // __CUDA_ARCH__ < 600
}

// Reciprocal square root functions
#if __CUDA_ARCH__ >= 530
__device__ inline float rsqrt_(__half x) {
  return hrsqrt(x);
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float rsqrt_(float x) {
  return rsqrtf(x);
}
__device__ inline double rsqrt_(double x) {
  return rsqrt(x);
}

/** CUDA kernel to compute channel sums.
 *  Sums and squares of sums are used to compute mean and variance.
 */
template <int block_size>
__global__ void channel_sums_kernel(
  int channel_height,
  int width,
  const DataType * __restrict__ data, int data_ldim,
        DataType * __restrict__ sums,
        DataType * __restrict__ sqsums) {

  // Indices
  const int tid = threadIdx.x;
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Initialize shared memory
  __shared__ DataType shared_sums[block_size];
  __shared__ DataType shared_sqsums[block_size];

  // Compute row sums in shared memory
  DataType private_sum = DataType(0);
  DataType private_sqsum = DataType(0);
  if (gidx < channel_height) {
    const int row = gidx + bidy * channel_height;
    for (int col = 0; col < width; ++col) {
      const auto& x = data[row + col * data_ldim];
      private_sum += x;
      private_sqsum += x * x;
    }
  }
  shared_sums[tid] = private_sum;
  shared_sqsums[tid] = private_sqsum;

  // Compute channel sum with shared memory reduction
  /// @todo unroll loops
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if(tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
      shared_sqsums[tid] += shared_sqsums[tid + stride];
    }
  }

  // Output channel sum to global memory
  if (tid == 0) {
    atomic_add(&sums[bidy], shared_sums[0]);
    atomic_add(&sqsums[bidy], shared_sqsums[0]);
  }

}

/** CUDA kernel to compute statistics.
 *  On input, global_mean and global_var are assumed to contain sums
 *  and squares of sums, respectively.
 */
__global__ void compute_statistics_kernel(
  int num_sums,
  int num_per_sum,
  DataType epsilon,
  DataType decay,
  DataType * __restrict__ global_mean,
  DataType * __restrict__ global_var,
  DataType * __restrict__ global_running_mean,
  DataType * __restrict__ global_running_var) {
  const DataType one = DataType(1);
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int i = gid; i < num_sums; i += num_threads) {

    // Compute mean and variance
    const auto& mean = global_mean[i] / num_per_sum;
    const auto& sqmean = global_var[i] / num_per_sum;
    auto var = num_per_sum * (sqmean - mean * mean) / (num_per_sum - 1);
    var = var > epsilon ? var : epsilon;
    global_mean[gid] = mean;
    global_var[gid] = var;

    // Compute running statistics
    auto& running_mean = global_running_mean[gid];
    auto& running_var = global_running_var[gid];
    running_mean = decay * running_mean + (one - decay) * mean;
    running_var = decay * running_var + (one - decay) * var;

  }

}

/** CUDA kernel to apply batch normalization. */
template <int block_size>
__global__ void batch_normalization_kernel(
  int channel_height,
  int width,
  const DataType * __restrict__ global_input, int input_ldim,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
  const DataType * __restrict__ global_bias,
        DataType * __restrict__ global_output, int output_ldim) {

  // Indices
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Copy batch normalization parameters to private memory
  const auto& mean = global_mean[bidy];
  const auto& var = global_var[bidy];
  const auto& scale = global_scale[bidy];
  const auto& bias = global_bias[bidy];

  // Get reciprocal of standard deviation
  const auto& inv_stdev = rsqrt_(var + epsilon);

  // Apply batch normalization
  if (gidx < channel_height) {
    const int row = gidx + bidy * channel_height;
    for (int col = 0; col < width; ++col) {
      const auto& x = global_input[row + col * input_ldim];
      const auto& xhat = (x - mean) * inv_stdev;
      const auto& y = scale * xhat + bias;
      global_output[row + col * output_ldim] = y;
    }
  }

}

/** CUDA kernel to compute gradients w.r.t. batch norm parameters. */
template <int block_size>
__global__ void backprop1_kernel(
  int channel_height,
  int width,
  const DataType * __restrict__ global_input,
  int input_ldim,
  const DataType * __restrict__ global_gradient_wrt_output,
  int gradient_wrt_output_ldim,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
        DataType * __restrict__ global_dscale,
        DataType * __restrict__ global_dbias,
        DataType * __restrict__ global_dmean,
        DataType * __restrict__ global_dvar) {

  // Indices
  const int tid = threadIdx.x;
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Initialize shared memory
  __shared__ DataType shared_dscale[block_size];
  __shared__ DataType shared_dbias[block_size];
  __shared__ DataType shared_dmean[block_size];
  __shared__ DataType shared_dvar[block_size];

  // Copy batch normalization parameters to private memory
  const auto& mean = global_mean[bidy];
  const auto& var = global_var[bidy];
  const auto& scale = global_scale[bidy];

  // Compute useful constants
  const DataType zero = DataType(0);
  const auto& inv_stdev = rsqrt_(var + epsilon);
  const auto& dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;

  // Compute row-wise gradient contributions in shared memory
  auto dscale = zero;
  auto dbias = zero;
  auto dmean = zero;
  auto dvar = zero;
  if (gidx < channel_height) {
    const int row = gidx + bidy * channel_height;
    for(int col = 0; col < width; ++col) {
      const auto& x = global_input[row + col * input_ldim];
      const auto& xhat = (x - mean) * inv_stdev;
      const auto& dy = global_gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      dscale += dy * xhat;
      dbias += dy;
      const auto& dxhat = dy * scale;
      dmean += - dxhat * inv_stdev;
      dvar += - dxhat * (x - mean) * dvar_factor;
    }
  }
  shared_dscale[tid] = dscale;
  shared_dbias[tid] = dbias;
  shared_dmean[tid] = dmean;
  shared_dvar[tid] = dvar;

  // Compute gradients with shared memory reduction
  // @todo unroll loops
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_dscale[tid] += shared_dscale[tid + stride];
      shared_dbias[tid] += shared_dbias[tid + stride];
      shared_dmean[tid] += shared_dmean[tid + stride];
      shared_dvar[tid] += shared_dvar[tid + stride];
    }
  }

  // Output channel sum to global memory
  if (tid == 0) {
    atomic_add(&global_dscale[bidy], shared_dscale[0]);
    atomic_add(&global_dbias[bidy], shared_dbias[0]);
    atomic_add(&global_dmean[bidy], shared_dmean[0]);
    atomic_add(&global_dvar[bidy], shared_dvar[0]);
  }

}

/** CUDA kernel to compute gradients w.r.t. input. */
template <int block_size>
__global__ void backprop2_kernel(
  int channel_height,
  int local_width,
  int global_width,
  const DataType * __restrict__ global_input,
  int input_ldim,
  const DataType * __restrict__ global_gradient_wrt_output,
  int gradient_wrt_output_ldim,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
  const DataType * __restrict__ global_dmean,
  const DataType * __restrict__ global_dvar,
        DataType * __restrict__ global_gradient_wrt_input,
  int gradient_wrt_input_ldim) {

  // Indices
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Copy batch normalization parameters to private memory
  const auto& mean = global_mean[bidy];
  const auto& var = global_var[bidy];
  const auto& scale = global_scale[bidy];
  const auto& dmean = global_dmean[bidy];
  const auto& dvar = global_dvar[bidy];

  // Compute useful constants
  const auto& inv_stdev = rsqrt_(var + epsilon);
  const auto& dmean_term = dmean / (global_width * channel_height);
  const auto& dvar_term = dvar * 2 / (global_width * channel_height - 1);

  // Apply batch normalization
  if (gidx < channel_height) {
    const int row = gidx + bidy * channel_height;
    for (int col = 0; col < local_width; ++col) {
      const auto& x = global_input[row + col * input_ldim];
      const auto& dy = global_gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      const auto& dxhat = dy * scale;
      auto dx = dxhat * inv_stdev;
      dx += dmean_term;
      dx += dvar_term * (x - mean);
      global_gradient_wrt_input[row + col * gradient_wrt_input_ldim] = dx;
    }
  }

}

} // namespace

namespace batch_normalization_cuda {

void channel_sums(int num_channels,
                  const AbsMat& data,
                  AbsMat& sums,
                  AbsMat& sqsums) {

#ifdef LBANN_DEBUG
  // Check that inputs are valid
  if (num_channels < 1) { LBANN_ERROR("non-positive number of channels"); }
  if (data.Height() % num_channels != 0) {
    LBANN_ERROR("number of channels does not divide input matrix height"); 
  }
  if (data.GetDevice() != El::Device::GPU
      || sums.GetDevice() != El::Device::GPU
      || sqsums.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices do not reside on GPU");
  }
#endif // LBANN_DEBUG  

  // Compute channel sums and squares of sums
  El::Zeros(sums, num_channels, 1);
  El::Zeros(sqsums, num_channels, 1);
  if (data.Height() > 0 && data.Width() > 0) {
    const int channel_height = data.Height() / num_channels;
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_height + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    channel_sums_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        channel_height, data.Width(),
        data.LockedBuffer(), data.LDim(),
        sums.Buffer(), sqsums.Buffer());
  }
}

void compute_statistics(int num_per_sum,
                        DataType epsilon,
                        DataType decay,
                        AbsMat& mean,
                        AbsMat& var,
                        AbsMat& running_mean,
                        AbsMat& running_var) {

#ifdef LBANN_DEBUG
  // Check that inputs are valid
  if (mean.Height() != var.Height()
      || mean.Height() != running_mean.Height()
      || mean.Height() != running_var.Height()
      || mean.Width() != 1 || var.Width() != 1
      || running_mean.Width() != 1 || running_var.Width() != 1) {
    LBANN_ERROR("invalid matrix dimensions");
  }
  if (mean.GetDevice() != El::Device::GPU
      || var.GetDevice() != El::Device::GPU
      || running_mean.GetDevice() != El::Device::GPU
      || running_var.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices do not reside on GPU");
  }
#endif // LBANN_DEBUG  

  // Compute statistics from sums
  const int block_dim = 256;
  const int grid_dim = (mean.Height() + block_dim - 1) / block_dim;
  if (num_per_sum > 1) {
    if (grid_dim > 0) {
      CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
      compute_statistics_kernel
        <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
          mean.Height(), num_per_sum, epsilon, decay,
          mean.Buffer(), var.Buffer(),
          running_mean.Buffer(), running_var.Buffer());
    }
  } else {
    El::Fill(var, DataType(1));
  }

}

void batch_normalization(const AbsMat& input,
                         const AbsMat& mean,
                         const AbsMat& var,
                         DataType epsilon,
                         const AbsMat& scale,
                         const AbsMat& bias,
                         AbsMat& output) {
  const int num_channels = mean.Height();

#ifdef LBANN_DEBUG
  // Check that inputs are valid
  if (num_channels < 1) { LBANN_ERROR("non-positive number of channels"); }
  if (input.Height() % num_channels != 0) {
    LBANN_ERROR("number of channels does not divide input matrix height"); 
  }
  if (mean.Height() != num_channels || var.Height() != num_channels
      || scale.Height() != num_channels || bias.Height() != num_channels
      || mean.Width() != 1 || var.Width() != 1
      || scale.Width() != 1 || bias.Width() != 1
      || input.Height() != output.Height()
      || input.Width() != output.Width()) {
    LBANN_ERROR("invalid matrix dimensions");
  }
  if (input.GetDevice() != El::Device::GPU
      || mean.GetDevice() != El::Device::GPU
      || var.GetDevice() != El::Device::GPU
      || scale.GetDevice() != El::Device::GPU
      || bias.GetDevice() != El::Device::GPU
      || output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices do not reside on GPU");
  }
#endif // LBANN_DEBUG  

  // Apply batch normalization
  if (input.Height() > 0 && input.Width() > 0) {
    const int channel_height = input.Height() / num_channels;
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_height + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    batch_normalization_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        channel_height, input.Width(),
        input.LockedBuffer(), input.LDim(),
        mean.LockedBuffer(), var.LockedBuffer(), epsilon,
        scale.LockedBuffer(), bias.LockedBuffer(),
        output.Buffer(), output.LDim());
  }

}

void backprop1(const AbsMat& input,
               const AbsMat& gradient_wrt_output,
               const AbsMat& mean,
               const AbsMat& var,
               DataType epsilon,
               const AbsMat& scale,
               AbsMat& dscale,
               AbsMat& dbias,
               AbsMat& dmean,
               AbsMat& dvar) {
  const int num_channels = mean.Height();

#ifdef LBANN_DEBUG
  // Check that inputs are valid
  if (num_channels < 1) { LBANN_ERROR("non-positive number of channels"); }
  if (input.Height() % num_channels != 0) {
    LBANN_ERROR("number of channels does not divide input matrix height"); 
  }
  if (mean.Height() != num_channels || var.Height() != num_channels
      || scale.Height() != num_channels
      || mean.Width() != 1 || var.Width() != 1 || scale.Width() != 1
      || input.Height() != gradient_wrt_output.Height()
      || input.Width() != gradient_wrt_output.Width()) {
    LBANN_ERROR("invalid matrix dimensions");
  }
  if (input.GetDevice() != El::Device::GPU
      || gradient_wrt_output.GetDevice() != El::Device::GPU
      || mean.GetDevice() != El::Device::GPU
      || var.GetDevice() != El::Device::GPU
      || scale.GetDevice() != El::Device::GPU
      || dscale.GetDevice() != El::Device::GPU
      || dbias.GetDevice() != El::Device::GPU
      || dmean.GetDevice() != El::Device::GPU
      || dvar.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices do not reside on GPU");
  }
#endif // LBANN_DEBUG  

  // Compute gradients w.r.t. batch norm parameters
  El::Zeros(dscale, num_channels, 1);
  El::Zeros(dbias, num_channels, 1);
  El::Zeros(dmean, num_channels, 1);
  El::Zeros(dvar, num_channels, 1);
  if (input.Height() > 0 && input.Width() > 0) {
    const int channel_height = input.Height() / num_channels;
    const int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_height + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    backprop1_kernel<block_size>
      <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
        channel_height, input.Width(),
        input.LockedBuffer(), input.LDim(),
        gradient_wrt_output.LockedBuffer(), gradient_wrt_output.LDim(),
        mean.LockedBuffer(), var.LockedBuffer(), epsilon,
        scale.LockedBuffer(), dscale.Buffer(), dbias.Buffer(),
        dmean.Buffer(), dvar.Buffer());
  }

}

void backprop2(int global_width,
               const AbsMat& input,
               const AbsMat& gradient_wrt_output,
               const AbsMat& mean,
               const AbsMat& var,
               DataType epsilon,
               const AbsMat& scale,
               const AbsMat& dmean,
               const AbsMat& dvar,
               AbsMat& gradient_wrt_input) {
  const int num_channels = mean.Height();

#ifdef LBANN_DEBUG
  // Check that inputs are valid
  if (num_channels < 1) { LBANN_ERROR("non-positive number of channels"); }
  if (input.Height() % num_channels != 0) {
    LBANN_ERROR("number of channels does not divide input matrix height"); 
  }
  if (mean.Height() != num_channels || var.Height() != num_channels
      || scale.Height() != num_channels
      || dmean.Height() != num_channels || dvar.Height() != num_channels
      || mean.Width() != 1 || var.Width() != 1 || scale.Width() != 1
      || dmean.Width() != 1 || dvar.Width() != 1
      || input.Height() != gradient_wrt_output.Height()
      || input.Height() != gradient_wrt_input.Height()
      || input.Width() != gradient_wrt_output.Width()
      || input.Width() != gradient_wrt_input.Width()) {
    LBANN_ERROR("invalid matrix dimensions");
  }
  if (input.GetDevice() != El::Device::GPU
      || gradient_wrt_output.GetDevice() != El::Device::GPU
      || mean.GetDevice() != El::Device::GPU
      || var.GetDevice() != El::Device::GPU
      || scale.GetDevice() != El::Device::GPU
      || dmean.GetDevice() != El::Device::GPU
      || dvar.GetDevice() != El::Device::GPU
      || gradient_wrt_input.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("matrices do not reside on GPU");
  }
#endif // LBANN_DEBUG  

  // Compute gradient w.r.t. input
  const int channel_height = input.Height() / num_channels;
  if (channel_height * global_width <= 1) {
    // El::Zero(gradient_wrt_input);
  } else {
    if (input.Height() > 0 && input.Width() > 0) {
      const int block_size = 256;
      dim3 block_dims, grid_dims;
      block_dims.x = block_size;
      grid_dims.x = (channel_height + block_size - 1) / block_size;
      grid_dims.y = num_channels;
      CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
      backprop2_kernel<block_size>
        <<<grid_dims, block_dims, 0, El::GPUManager::Stream()>>>(
          channel_height, input.Width(), global_width,
          input.LockedBuffer(), input.LDim(),
          gradient_wrt_output.LockedBuffer(), gradient_wrt_output.LDim(),
          mean.LockedBuffer(), var.LockedBuffer(), epsilon,
          scale.LockedBuffer(), dmean.LockedBuffer(), dvar.LockedBuffer(),
          gradient_wrt_input.Buffer(), gradient_wrt_input.LDim());
    }
  }

}

} // namespace batch_normalization
} // namespace lbann
