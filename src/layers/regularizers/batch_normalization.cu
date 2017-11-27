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
//
// batch_normalization.cu - GPU helper routines for batch normalization layer
////////////////////////////////////////////////////////////////////////////////

#include "math.h"
#include <iostream>
#include "lbann/layers/regularizers/batch_normalization_cuda.hpp"
#include "lbann/utils/exception.hpp"

// Macros to check CUDA calls
#define FORCE_CHECK_CUDA(cuda_call)                                     \
  do {                                                                  \
    const cudaError_t status = cuda_call;                               \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("CUDA error");                       \
    }                                                                   \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDA(cuda_call) FORCE_CHECK_CUDA(cuda_call)
#else
#define CHECK_CUDA(cuda_call) cuda_call
#endif // #ifdef LBANN_DEBUG

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
__device__ inline float reciprocal_square_root(__half x) {
  return hrsqrt(x);
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float reciprocal_square_root(float x) {
  return rsqrtf(x);
}
__device__ inline double reciprocal_square_root(double x) {
  return rsqrt(x);
}

namespace lbann {
namespace batch_normalization_cuda {

template <typename DataType, int block_size>
__global__ void channel_sums_and_sqsums_kernel(
  int height,
  int width,
  int channel_size,
  const DataType * __restrict__ global_data,
        DataType * __restrict__ global_sums,
        DataType * __restrict__ global_sqsums) {

  // Indices
  const int tid = threadIdx.x;
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Initialize shared memory
  __shared__ DataType shared_sums[block_size];
  __shared__ DataType shared_sqsums[block_size];

  // Compute row sums in shared memory
  DataType sum = DataType(0);
  DataType sqsum = DataType(0);
  if(gidx < channel_size) {
    const int row = gidx + bidy * channel_size;
    for(int col = 0; col < width; ++col) {
      const DataType x = global_data[row + col * height];
      sum += x;
      sqsum += x * x;
    }
  }
  shared_sums[tid] = sum;
  shared_sqsums[tid] = sqsum;

  // Compute channel sum with shared memory reduction
  // TODO: unroll loops
  for(int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if(tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
      shared_sqsums[tid] += shared_sqsums[tid + stride];
    }
  }

  // Output channel sum to global memory
  if(tid == 0) {
    atomic_add(&global_sums[bidy], shared_sums[0]);
    atomic_add(&global_sqsums[bidy], shared_sqsums[0]);
  }

}

template <typename DataType>
void channel_sums_and_sqsums(int height,
                             int width,
                             int num_channels,
                             const DataType *data_d,
                                   DataType *sums_d,
                                   DataType *sqsums_d,
                             cudaStream_t stream) {
  
  // CUDA block size
  const int block_size = 256;

  // Clear GPU memory
  CHECK_CUDA(cudaMemsetAsync(sums_d, 0, num_channels * sizeof(DataType), stream));
  CHECK_CUDA(cudaMemsetAsync(sqsums_d, 0, num_channels * sizeof(DataType), stream));

  // Return if there is no input data
  if(width <= 0) return;

  // Launch CUDA kernel to compute sums and sums of squares
  const int channel_size = height / num_channels;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (channel_size + block_size - 1) / block_size;
  grid_dims.y = num_channels;
  channel_sums_and_sqsums_kernel<DataType,block_size>
    <<<grid_dims, block_dims, 0, stream>>>
    (height, width, channel_size, data_d, sums_d, sqsums_d);

}

template <typename DataType>
__global__ void sums_to_statistics_kernel(
  int num_entries,
  DataType samples_per_sum,
  DataType decay,
  DataType * __restrict__ global_mean,
  DataType * __restrict__ global_var,
  DataType * __restrict__ global_running_mean,
  DataType * __restrict__ global_running_var) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  while(gid < num_entries) {

    // Compute statistics
    const DataType mean = global_mean[gid] / samples_per_sum;
    const DataType sqmean = global_var[gid] / samples_per_sum;
    DataType var = sqmean - mean * mean;
    var = var > DataType(0) ? var : DataType(0);
    var *= samples_per_sum / (samples_per_sum - DataType(1));
    global_mean[gid] = mean;
    global_var[gid] = var;

    // Compute running statistics
    DataType& running_mean = global_running_mean[gid];
    DataType& running_var = global_running_var[gid];
    running_mean = decay * running_mean + (DataType(1) - decay) * mean;
    running_var = decay * running_var + (DataType(1) - decay) * var;
    
    gid += blockDim.x * gridDim.x;
  }
}

template <typename DataType>
void sums_to_statistics(int num_entries,
                        int samples_per_sum,
                        DataType decay,
                        DataType *mean_d,
                        DataType *var_d,
                        DataType *running_mean_d,
                        DataType *running_var_d,
                        cudaStream_t stream) {
  dim3 block_dims, grid_dims;
  block_dims.x = 256;
  grid_dims.x = (num_entries + block_dims.x - 1) / block_dims.x;
  sums_to_statistics_kernel<DataType>
    <<<grid_dims, block_dims, 0, stream>>>
    (num_entries, (DataType)samples_per_sum, decay,
     mean_d, var_d, running_mean_d, running_var_d);
}

template <typename DataType, int block_size>
__global__ void batch_normalization_kernel(
  int height,
  int width,
  int channel_size,
  const DataType * __restrict__ global_prev_activations,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
  const DataType * __restrict__ global_bias,
        DataType * __restrict__ global_activations) {

  // Indices
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Copy batch normalization parameters to private memory
  const DataType mean = global_mean[bidy];
  const DataType var = global_var[bidy];
  const DataType scale = global_scale[bidy];
  const DataType bias = global_bias[bidy];

  // Get reciprocal of standard deviation
  const DataType inv_stdev = reciprocal_square_root(var + epsilon);

  // Apply batch normalization
  if(gidx < channel_size) {
    const int row = gidx + bidy * channel_size;
    for(int col = 0; col < width; ++col) {
      const DataType x = global_prev_activations[row + col * height];
      const DataType xhat = (x - mean) * inv_stdev;
      const DataType y = scale * xhat + bias;
      global_activations[row + col * height] = y;
    }
  }

}

template <typename DataType>
void batch_normalization(int height,
                         int width,
                         int num_channels,
                         const DataType *prev_activations_d,
                         const DataType *mean_d,
                         const DataType *var_d,
                         DataType epsilon,
                         const DataType *scale_d,
                         const DataType *bias_d,
                               DataType *activations_d,
                         cudaStream_t stream) {

  // CUDA block size
  const int block_size = 256;

  // Return if there is no input data
  if(width <= 0) return;

  // Launch CUDA kernel to apply batch normalization
  const int channel_size = height / num_channels;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (channel_size + block_size - 1) / block_size;
  grid_dims.y = num_channels;
  batch_normalization_kernel<DataType,block_size>
    <<<grid_dims, block_dims, 0, stream>>>
    (height, width, channel_size,
     prev_activations_d,
     mean_d, var_d, epsilon,
     scale_d, bias_d,
     activations_d);

}

template <typename DataType, int block_size>
__global__ void batch_normalization_backprop1_kernel(
  int height,
  int width,
  int channel_size,
  const DataType * __restrict__ global_prev_activations,
  const DataType * __restrict__ global_prev_error_signal,
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
  const DataType mean = global_mean[bidy];
  const DataType var = global_var[bidy];
  const DataType scale = global_scale[bidy];

  // Compute useful constants
  const DataType inv_stdev = reciprocal_square_root(var + epsilon);
  const DataType dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;

  // Compute row-wise gradient contributions in shared memory
  DataType dscale = DataType(0);
  DataType dbias = DataType(0);
  DataType dmean = DataType(0);
  DataType dvar = DataType(0);
  if(gidx < channel_size) {
    const int row = gidx + bidy * channel_size;
    for(int col = 0; col < width; ++col) {
      const DataType x = global_prev_activations[row + col * height];
      const DataType xhat = (x - mean) * inv_stdev;
      const DataType dy = global_prev_error_signal[row + col * height];
      dscale += dy * xhat;
      dbias += dy;
      const DataType dxhat = dy * scale;
      dmean += - dxhat * inv_stdev;
      dvar += - dxhat * (x - mean) * dvar_factor;
    }
  }
  shared_dscale[tid] = dscale;
  shared_dbias[tid] = dbias;
  shared_dmean[tid] = dmean;
  shared_dvar[tid] = dvar;

  // Compute gradients with shared memory reduction
  // TODO: unroll loops
  for(int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if(tid < stride) {
      shared_dscale[tid] += shared_dscale[tid + stride];
      shared_dbias[tid] += shared_dbias[tid + stride];
      shared_dmean[tid] += shared_dmean[tid + stride];
      shared_dvar[tid] += shared_dvar[tid + stride];
    }
  }

  // Output channel sum to global memory
  if(tid == 0) {
    atomic_add(&global_dscale[bidy], shared_dscale[0]);
    atomic_add(&global_dbias[bidy], shared_dbias[0]);
    atomic_add(&global_dmean[bidy], shared_dmean[0]);
    atomic_add(&global_dvar[bidy], shared_dvar[0]);
  }

}

template <typename DataType>
void batch_normalization_backprop1(int height,
                                   int width,
                                   int num_channels,
                                   const DataType *prev_activations_d,
                                   const DataType *prev_error_signal_d,
                                   const DataType *mean_d,
                                   const DataType *var_d,
                                   DataType epsilon,
                                   const DataType *scale_d,
                                         DataType *dscale_d,
                                         DataType *dbias_d,
                                         DataType *dmean_d,
                                         DataType *dvar_d,
                                   cudaStream_t stream) {
  
  // CUDA block size
  const int block_size = 256;

  // Clear GPU memory
  CHECK_CUDA(cudaMemsetAsync(dscale_d, 0, num_channels * sizeof(DataType), stream));
  CHECK_CUDA(cudaMemsetAsync(dbias_d, 0, num_channels * sizeof(DataType), stream));
  CHECK_CUDA(cudaMemsetAsync(dmean_d, 0, num_channels * sizeof(DataType), stream));
  CHECK_CUDA(cudaMemsetAsync(dvar_d, 0, num_channels * sizeof(DataType), stream));

  // Return if there is no input data
  if(width <= 0) return;

  // Launch CUDA kernel for first phase of batch normalization backward propagation
  const int channel_size = height / num_channels;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (channel_size + block_size - 1) / block_size;
  grid_dims.y = num_channels;
  batch_normalization_backprop1_kernel<DataType,block_size>
    <<<grid_dims, block_dims, 0, stream>>>
    (height, width, channel_size,
     prev_activations_d, prev_error_signal_d,
     mean_d, var_d, epsilon, scale_d,
     dscale_d, dbias_d, dmean_d, dvar_d);

}

template <typename DataType, int block_size>
__global__ void batch_normalization_backprop2_kernel(
  int height,
  int local_width,
  int global_width,
  int channel_size,
  const DataType * __restrict__ global_prev_activations,
  const DataType * __restrict__ global_prev_error_signal,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
  const DataType * __restrict__ global_dmean,
  const DataType * __restrict__ global_dvar,
        DataType * __restrict__ global_error_signal) {

  // Indices
  const int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int bidy = blockIdx.y;

  // Copy batch normalization parameters to private memory
  const DataType mean = global_mean[bidy];
  const DataType var = global_var[bidy];
  const DataType scale = global_scale[bidy];
  const DataType dmean = global_dmean[bidy];
  const DataType dvar = global_dvar[bidy];

  // Compute useful constants
  const DataType inv_stdev = reciprocal_square_root(var + epsilon);
  const DataType dmean_term = dmean / (global_width * channel_size);
  const DataType dvar_term = dvar * 2 / (global_width * channel_size - 1);

  // Apply batch normalization
  if(gidx < channel_size) {
    const int row = gidx + bidy * channel_size;
    for(int col = 0; col < local_width; ++col) {
      const DataType x = global_prev_activations[row + col * height];
      const DataType dy = global_prev_error_signal[row + col * height];
      const DataType dxhat = dy * scale;
      DataType dx = dxhat * inv_stdev;
      dx += dmean_term;
      dx += dvar_term * (x - mean);
      global_error_signal[row + col * height] = dx;
    }
  }

}

template <typename DataType>
void batch_normalization_backprop2(int height,
                                   int local_width,
                                   int global_width,
                                   int num_channels,
                                   const DataType *prev_activations_d,
                                   const DataType *prev_error_signal_d,
                                   const DataType *mean_d,
                                   const DataType *var_d,
                                   DataType epsilon,
                                   const DataType *scale_d,
                                   const DataType *dmean_d,
                                   const DataType *dvar_d,
                                         DataType *error_signal_d,
                                   cudaStream_t stream) {
  
  // CUDA block size
  const int block_size = 256;

  // Return if there is no input data
  if(local_width <= 0) return;

  // Launch CUDA kernel for second phase of batch normalization backward propagation
  const int channel_size = height / num_channels;
  dim3 block_dims, grid_dims;
  block_dims.x = block_size;
  grid_dims.x = (channel_size + block_size - 1) / block_size;
  grid_dims.y = num_channels;
  batch_normalization_backprop2_kernel<DataType,block_size>
    <<<grid_dims, block_dims, 0, stream>>>
    (height, local_width, global_width, channel_size,
     prev_activations_d, prev_error_signal_d,
     mean_d, var_d, epsilon, scale_d, dmean_d, dvar_d,
     error_signal_d);

}

// Explicit instantiation
template
void channel_sums_and_sqsums<float>(int height,
                                    int width,
                                    int num_channels,
                                    const float *data_d,
                                    float *sums_d,
                                    float *sqsums_d,
                                    cudaStream_t stream);
template
void sums_to_statistics<float>(int num_entries,
                               int entries_per_sum,
                               float decay,
                               float *mean_d,
                               float *var_d,
                               float *running_mean_d,
                               float *running_var_d,
                               cudaStream_t stream);
template
void batch_normalization<float>(int height,
                                int width,
                                int num_channels,
                                const float *prev_activations_d,
                                const float *mean_d,
                                const float *var_d,
                                float epsilon,
                                const float *scale_d,
                                const float *bias_d,
                                float *activations_d,
                                cudaStream_t stream);
template
void batch_normalization_backprop1<float>(int height,
                                          int width,
                                          int num_channels,
                                          const float *prev_activations_d,
                                          const float *prev_error_signal_d,
                                          const float *mean_d,
                                          const float *var_d,
                                          float epsilon,
                                          const float *scale_d,
                                          float *dscale_d,
                                          float *dbias_d,
                                          float *dmean_d,
                                          float *dvar_d,
                                          cudaStream_t stream);
template
void batch_normalization_backprop2<float>(int height,
                                          int local_width,
                                          int global_width,
                                          int num_channels,
                                          const float *prev_activations_d,
                                          const float *prev_error_signal_d,
                                          const float *mean_d,
                                          const float *var_d,
                                          float epsilon,
                                          const float *scale_d,
                                          const float *dmean_d,
                                          const float *dvar_d,
                                          float *error_signal_d,
                                          cudaStream_t stream);
template
void channel_sums_and_sqsums<double>(int height,
                                     int width,
                                     int num_channels,
                                     const double *data_d,
                                     double *sums_d,
                                     double *sqsums_d,
                                     cudaStream_t stream);
template
void sums_to_statistics<double>(int num_entries,
                                int entries_per_sum,
                                double decay,
                                double *mean_d,
                                double *var_d,
                                double *running_mean_d,
                                double *running_var_d,
                                cudaStream_t stream);
template
void batch_normalization<double>(int height,
                                 int width,
                                 int num_channels,
                                 const double *prev_activations_d,
                                 const double *mean_d,
                                 const double *var_d,
                                 double epsilon,
                                 const double *scale_d,
                                 const double *bias_d,
                                 double *activations_d,
                                 cudaStream_t stream);
template
void batch_normalization_backprop1<double>(int height,
                                           int width,
                                           int num_channels,
                                           const double *prev_activations_d,
                                           const double *prev_error_signal_d,
                                           const double *mean_d,
                                           const double *var_d,
                                           double epsilon,
                                           const double *scale_d,
                                           double *dscale_d,
                                           double *dbias_d,
                                           double *dmean_d,
                                           double *dvar_d,
                                           cudaStream_t stream);
template
void batch_normalization_backprop2<double>(int height,
                                           int local_width,
                                           int global_width,
                                           int num_channels,
                                           const double *prev_activations_d,
                                           const double *prev_error_signal_d,
                                           const double *mean_d,
                                           const double *var_d,
                                           double epsilon,
                                           const double *scale_d,
                                           const double *dmean_d,
                                           const double *dvar_d,
                                           double *error_signal_d,
                                           cudaStream_t stream);

} // namespace batch_normalization
} // namespace lbann
