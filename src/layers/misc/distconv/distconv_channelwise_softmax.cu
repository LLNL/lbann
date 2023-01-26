////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_LAYERS_MISC_CHANNELWISE_SOFTMAX_INSTANTIATE

#ifdef LBANN_HAS_DISTCONV
namespace distconv{
namespace{

using Size3 = gpu_lib::array<size_t,3>;

/** @brief Max functor */
template <class T>
struct max_op {
  __device__ __forceinline__
  DataType operator()(const T& x1, const T& x2) const {
    return gpu_lib::max(x1, x2);
  }
};

} // namespace <anon>

// =========================================================
// Forward prop
// =========================================================

namespace {

/** @brief Max reduction over last dimension of 3D tensor.
 *
 *  Each CUDA block computes the max over a subset of tensor entries
 *  in @c vals and outputs the result to @c maxvals. This should be
 *  repeated multiple times to fully reduce the last tensor dimension.
 *
 *  Block dimensions: bdimx x 1 x 1
 *
 *  Grid dimensions: (vals_dims[2] / bdimx) x vals_dims[1] x vals_dims[0]
 *
 *  maxvals: vals_dims[0] x vals_dims[1] x (vals_dims[2] / bdimx)
 */
template <typename TensorDataType, size_t bdimx>
__global__ void fp_max_kernel(
  Size3 vals_dims,
  const TensorDataType* __restrict__ vals_buffer,
  Size3 vals_strides,
  TensorDataType* __restrict__ maxvals_buffer,
  Size3 maxvals_strides) {

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x;
  const size_t bidx = blockIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  for (size_t k = gidz; k < vals_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < vals_dims[1]; j += nthreadsy) {

      // Find largest value for each thread
      TensorDataType maxval{-gpu_lib::infinity<TensorDataType>()};
      for (size_t i = gidx; i < vals_dims[2]; i += nthreadsx) {
        const auto& val = vals_buffer[k * vals_strides[0]
                                      + j * vals_strides[1]
                                      + i * vals_strides[2]];
        maxval = gpu_lib::max(maxval, val);
      }

      // Find largest value for each block
      maxval = gpu_lib::block_reduce<bdimx,bdimy,bdimz,TensorDataType,max_op<TensorDataType>>(maxval);
      if (tid == 0) {
        const auto& pos = (k * maxvals_strides[0]
                           + j * maxvals_strides[1]
                           + bidx * maxvals_strides[2]);
        maxvals_buffer[pos] = maxval;
      }

    }
  }

}

/** Compute softmax.
 *
 *  y_i = exp(x_i-shift) / denom
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (input_dims[2] / bdimx) x (input_dims[1] / bdimy) x (input_dims[0] / bdimz)
 *
 *  shifts and denoms are fully-packed 2D tensors with dimensions of
 *  input_dims[0] x input_dims[1].
 */
template <typename TensorDataType>
__global__ void fp_output_kernel(
  Size3 input_dims,
  const TensorDataType* __restrict__ input_buffer,
  Size3 input_strides,
  TensorDataType* __restrict__ output_buffer,
  Size3 output_strides,
  const TensorDataType* __restrict__ shifts,
  const TensorDataType* __restrict__ denoms) {

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;
  for (size_t k = gidz; k < input_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < input_dims[1]; j += nthreadsy) {
      const auto& shift = shifts[j + k*input_dims[1]];
      const auto& denom = denoms[j + k*input_dims[1]];
      for (size_t i = gidx; i < input_dims[2]; i += nthreadsx) {
        const auto& x = input_buffer[k * input_strides[0]
                                     + j * input_strides[1]
                                     + i * input_strides[2]];
        auto& y = output_buffer[k * output_strides[0]
                                + j * output_strides[1]
                                + i * output_strides[2]];
        y = gpu_lib::exp(x-shift) / denom;
      }
    }
  }
}

} // Namespace <anon>



// =========================================================
// Backprop
// =========================================================

namespace {
/** Compute dot product between output and gradient w.r.t. output.
 *
 *  Block dimensions: bdimx x 1 x 1
 *
 *  Grid dimensions: (output_dims[2] / bdimx) x output_dims[1] x output_dims[0]
 *
 *  y_dot_dy is a fully-packed 2D tensor with dimensions of
 *  output_dims[0] x output_dims[1].
 */
template <typename TensorDataType, size_t bdimx>
__global__ void bp_y_dot_dy_kernel(
  Size3 output_dims,
  const TensorDataType* __restrict__ output_buffer,
  Size3 output_strides,
  const TensorDataType* __restrict__ output_grad_buffer,
  Size3 output_grad_strides,
  TensorDataType* __restrict__ y_dot_dy) {

  // Indices and dimensions
  constexpr size_t bdimy = 1;
  constexpr size_t bdimz = 1;
  const size_t tid = threadIdx.x;
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;

  for (size_t k = gidz; k < output_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < output_dims[1]; j += nthreadsy) {

      // Compute contribution from each thread
      TensorDataType _y_dot_dy{0.};
      for (size_t i = gidx; i < output_dims[2]; i += nthreadsx) {
        const auto& y = output_buffer[k * output_strides[0]
                                      + j * output_strides[1]
                                      + i * output_strides[2]];
        const auto& dy = output_grad_buffer[k * output_grad_strides[0]
                                            + j * output_grad_strides[1]
                                            + i * output_grad_strides[2]];
        _y_dot_dy += y * dy;
      }

      // Compute contribution from each block
      _y_dot_dy = gpu_lib::block_reduce<bdimx,bdimy,bdimz>(_y_dot_dy);
      if (tid == 0) {
        gpu_lib::atomic_add(&y_dot_dy[j+k*output_dims[1]], _y_dot_dy);
      }

    }
  }

}

/** Compute gradient w.r.t. input.
 *
 *  dL/dx_i = y_i * ( dL/dy_i - dot(y,dL/dy) )
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (output_dims[2] / bdimx) x (output_dims[1] / bdimy) x (output_dims[0] / bdimz)
 *
 *  y_dot_dy is a fully-packed 2D tensor with dimensions of
 *  output_dims[0] x output_dims[1].
 */
template <typename TensorDataType>
__global__ void bp_input_grad_kernel(
  Size3 output_dims,
  const TensorDataType* __restrict__ output_buffer,
  Size3 output_strides,
  const TensorDataType* __restrict__ output_grad_buffer,
  Size3 output_grad_strides,
  TensorDataType* __restrict__ input_grad_buffer,
  Size3 input_grad_strides,
  const TensorDataType* __restrict__ y_dot_dy) {

  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  const size_t nthreadsz = blockDim.z * gridDim.z;
  for (size_t k = gidz; k < output_dims[0]; k += nthreadsz) {
    for (size_t j = gidy; j < output_dims[1]; j += nthreadsy) {
      const auto& _y_dot_dy = y_dot_dy[j + k*output_dims[1]];
      for (size_t i = gidx; i < output_dims[2]; i += nthreadsx) {
        const auto& y = output_buffer[k * output_strides[0]
                                      + j * output_strides[1]
                                      + i * output_strides[2]];
        const auto& dy = output_grad_buffer[k * output_grad_strides[0]
                                            + j * output_grad_strides[1]
                                            + i * output_grad_strides[2]];
        auto& dx = input_grad_buffer[k * input_grad_strides[0]
                                    + j * input_grad_strides[1]
                                    + i * input_grad_strides[2]];
        dx = y * (dy - _y_dot_dy);
      }
    }
  }

}

}  // namespace <anon>

  
  template<typename Backend, typename DataType>
  template<typename Allocator>
  int
  ChannelwiseSoftmax<Backend, DataType>
  ::forward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_0,
            tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output){

    if (input_0.get_local_size() == 0 || output.get_local_size()){
      return 1; // no op for empty inputs
    }

    const auto& input_0_dims = input_0.get_local_shape();
    
    const auto num_channels = input_0_dims[2];
    const auto local_mini_batch_size = input_0_dims[3];
    const auto mat_channel_size = input_0_dims[0] * input_0_dims[1];
    const auto mat_stride = num_channels * mat_channel_size;

    // Convert to Hydrogen matrices for kernel launch

    using LocalMat = El::Matrix<DataType, El::Device::GPU>;

    LocalMat local_input(mat_stride,
                        local_mini_batch_size,
                        input_0.get_buffer(),
                        mat_stride);

    LocalMat local_output(mat_stride,
                          local_mini_batch_size,
                          output.get_buffer(),
                          mat_stride);
    {
      using namespace hydrogen;
      using Size3 = gpu_lib::array<size_t,3>;

      auto  multisync = MakeMultiSync(El::SyncInfoFromMatrix(local_input),
                                      El::SyncInfoFromMatrix(local_output));

      LocalMat local_shifts;
      if (!local_input.IsEmpty()) {
        constexpr size_t block_size = 256;
        dim3 block_dims, grid_dims;
        block_dims.x = block_size;
        grid_dims.x = (channel_size + block_size - 1) / block_size;
        grid_dims.y = num_channels;
        grid_dims.z = local_mini_batch_size;
        gpu_lib::clip_grid_dims(grid_dims);
        LocalMat maxvals(grid_dims.x * num_channels, local_mini_batch_size);
        hydrogen::gpu::LaunchKernel(
          fp_max_kernel<TensorDataType, block_size>,
          grid_dims, block_dims, 0, multisync,
          Size3{local_mini_batch_size, num_channels, channel_size},
          local_input.LockedBuffer(),
          Size3{static_cast<size_t>(local_input.LDim()), channel_size, 1},
          maxvals.Buffer(),
          Size3{static_cast<size_t>(maxvals.LDim()), grid_dims.x, 1});
        while (grid_dims.x > 1) {
          const size_t prev_dim = grid_dims.x;
          grid_dims.x = (prev_dim + block_size - 1) / block_size;
          const LocalMat prev_maxvals(std::move(maxvals));
          maxvals.Resize(grid_dims.x * num_channels, local_mini_batch_size);
          hydrogen::gpu::LaunchKernel(
            fp_max_kernel<TensorDataType, block_size>,
            grid_dims, block_dims, 0, multisync,
            Size3{local_mini_batch_size, num_channels, prev_dim},
            prev_maxvals.LockedBuffer(),
            Size3{static_cast<size_t>(prev_maxvals.LDim()), prev_dim, 1},
            maxvals.Buffer(),
            Size3{static_cast<size_t>(maxvals.LDim()), grid_dims.x, 1});
        }
        local_shifts = std::move(maxvals);
      }

      // Compute softmax denominators
      LocalMat local_denoms(num_channels, local_mini_batch_size);
      El::Zero(local_denoms);
      if (!local_input.IsEmpty()) {
        constexpr size_t block_size = 256;
        dim3 block_dims, grid_dims;
        block_dims.x = block_size;
        grid_dims.x = (channel_size + block_size - 1) / block_size;
        grid_dims.y = num_channels;
        grid_dims.z = local_mini_batch_size;
        gpu_lib::clip_grid_dims(grid_dims);
        hydrogen::gpu::LaunchKernel(
          fp_denom_kernel<TensorDataType, block_size>,
          grid_dims, block_dims, 0, multisync,
          Size3{local_mini_batch_size, num_channels, channel_size},
          local_input.LockedBuffer(),
          Size3{static_cast<size_t>(local_input.LDim()), channel_size, 1},
          local_shifts.LockedBuffer(),
          local_denoms.Buffer());
      }

      // Compute softmax
      if (!local_input.IsEmpty()) {
        constexpr size_t block_size = 256;
        dim3 block_dims, grid_dims;
        block_dims.x = block_size;
        grid_dims.x = (channel_size + block_size - 1) / block_size;
        grid_dims.y = num_channels;
        grid_dims.z = local_mini_batch_size;
        gpu_lib::clip_grid_dims(grid_dims);
        hydrogen::gpu::LaunchKernel(
          fp_output_kernel<TensorDataType>,
          grid_dims, block_dims, 0, multisync,
          Size3{local_mini_batch_size, num_channels, channel_size},
          local_input.LockedBuffer(),
          Size3{static_cast<size_t>(local_input.LDim()), channel_size, 1},
          local_output.Buffer(),
          Size3{static_cast<size_t>(local_output.LDim()), channel_size, 1},
          local_shifts.LockedBuffer(),
          local_denoms.LockedBuffer());
      }
 
    }  // namespace hydrogen
    return 1;        
  }

  template<typename Backend, typename DataType>
  template<typename Allocator>
  int
  ChannelwiseSoftmax<Backend, DataType>
  ::backward(const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_0,
             const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &output_grad,
             tensor::Tensor<DataType, tensor::LocaleMPI, Allocator> &input_grad_0){

    return 1;        
  }

// =========================================================
// Explicit template instantiation
// =========================================================
}  // namespace distconv
#endif // LBANN_HAS_DISTCONV