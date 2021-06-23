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

#define LBANN_SCATTER_LAYER_INSTANTIATE
#include "lbann/layers/transform/scatter.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

using Dim2 = gpu_lib::array<size_t, 2>;
using Dim3 = gpu_lib::array<size_t, 3>;

/** @brief Kernel for scattering a 2D tensor along dim 1
 *
 *  output(j,indices(j,i)) = values(j,i)
 *
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (values_dim[1] / bdimx) x (values_dim[0] / bdimy) x 1
 */
template <typename T>
__global__ void scatter2d_kernel(
  const T* __restrict__ indices,
  Dim2 indices_strides,
  const T* __restrict__ values,
  Dim3 values_dims,
  Dim3 values_strides,
  T* __restrict__ output,
  Dim3 output_dims,
  Dim3 output_strides) {

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;


  auto mini_batch_size = output_dims[0];
  auto num_rows = output_dims[1];
  auto num_out_columns = output_dims[2];
  auto num_value_columns = values_dims[2];


  for (size_t batch = gidz; batch < mini_batch_size; batch+=nthreadsz){
    for(size_t row = gidy; row < num_rows; row+=nthreadsy){
      for (size_t i = gidx; i < num_value_columns; i+=nthreadsx){
        const auto ind = static_cast<El::Int>(
          gpu_lib::floor(
            indices[batch*indices_strides[0] + i*indices_strides[1]]));
        if (0<=ind && ind < static_cast<El::Int>(num_out_columns)){
          const auto& x = values[batch*values_strides[0] + row*values_strides[1] + i*values_strides[2]];
          auto &y = output[batch*output_strides[0] + row*output_strides[1] + ind*output_strides[2]];
          gpu_lib::atomic_add(&y, x);
        }
      }
    }
  }
}

/** @brief Kernel for gathering a 2D tensor along dim 1
 *
 *  output(j,i) = values(j,indices(j,i))
 *
 *  Block dimensions: bdimx x bdimy x 1
 *
 *  Grid dimensions: (output_dim[1] / bdimx) x (output_dim[0] / bdimy) x 1
 */
template <typename T>
__global__ void gather2d_kernel(
  const T* __restrict__ indices,
  Dim2 indices_strides,
  const T* __restrict__ values,
  Dim3 values_dims,
  Dim3 values_strides,
  T* __restrict__ output,
  Dim3 output_dims,
  Dim3 output_strides) {

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;
  const size_t nthreadsz = gridDim.z * blockDim.z;

  auto mini_batch_size = output_dims[0];
  auto num_rows = output_dims[1];
  auto num_out_columns = output_dims[2];
  auto num_value_columns = values_dims[2];


  for (size_t j=gidz; j<mini_batch_size; j+=nthreadsz) { //Every sample in MB 
    for (size_t k=gidx; k<output_dims[1]; k+=nthreadsx) { // Every row
      for (size_t i = gidy; i < output_dims[2]; i+=nthreadsy){ // Every column
        const auto ind = static_cast<El::Int>(
          gpu_lib::floor(
            indices[j*indices_strides[0] + i*indices_strides[1]]));
        
        auto& y = output[j*output_strides[0] + k*output_strides[1] + i * output_strides[2]];
        
        if (0<=ind && ind<static_cast<El::Int>(values_dims[2])) {
          y = values[j*values_strides[0] + k*values_strides[1] + ind*values_strides[2]];
        }
        else {
          y = T{0.f};
        }
      }
    }
  }
}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void scatter_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Local matrices
  const auto& local_values = this->get_local_prev_activations(0);
  const auto& local_indices = this->get_local_prev_activations(1);
  auto& local_output = this->get_local_activations();

  const auto& input_dims_ = this->get_input_dims();
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> input_dims(input_dims_.begin(), input_dims_.end());
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());

  const size_t local_mini_batch_size = local_indices.Width();

  const size_t values_size = (input_dims.size()>1) ? input_dims[1] : this->get_input_size(0);
  const size_t output_size = (input_dims.size()>1) ?  this->get_output_dims()[1] : this->get_output_size();
  const size_t num_rows = (input_dims.size()>1) ? input_dims[0] : 1;
  const size_t value_stride_2 = (input_dims.size()>1) ? values_size : 0;
  const size_t output_stride_2 = (input_dims.size()>1) ? output_size : 0;



  // Scatter into output matrix
  El::Zero(local_output);
  if (!local_values.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_output),
                                       gpu::get_sync_info(local_values),
                                       gpu::get_sync_info(local_indices));
    constexpr size_t block_size_x = 32;
    constexpr size_t block_size_y = 8;
    
    dim3 block_dims, grid_dims;
    block_dims.x = block_size_x;
    block_dims.y = block_size_y;
    block_dims.z = 1;

    grid_dims.x = (values_size + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (num_rows + block_dims.y - 1) / block_dims.y;
    grid_dims.z = (local_mini_batch_size + block_dims.z - 1) / block_dims.z; 



    std::cout << "Num Rows " << num_rows 
              << " Num Y columns " << output_size 
              << " Num X columns " << values_size
              << std::endl;

    std::cout << " Strides : \t" << local_indices.LDim() <<
                  "\tValues LDIM: \t" << local_values.LDim() << 
                  "\t Values Stride 2\t" << value_stride_2 <<
                  "\tOutput LDIM: \t" <<local_output.LDim() << 
                  "\t Output Stride 2\t" << output_stride_2 << std::endl; 
    hydrogen::gpu::LaunchKernel(
      scatter2d_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_indices.LockedBuffer(),
      Dim2{static_cast<size_t>(local_indices.LDim()), 1},
      local_values.LockedBuffer(),
      Dim3{local_mini_batch_size, num_rows, values_size},
      Dim3{static_cast<size_t>(local_values.LDim()), value_stride_2, 1},
      local_output.Buffer(),
      Dim3{local_mini_batch_size, num_rows, output_size},
      Dim3{static_cast<size_t>(local_output.LDim()), output_stride_2, 1});
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void scatter_layer<TensorDataType, Layout, Device>::bp_compute() {

  // Local matrices
  const auto& local_indices = this->get_local_prev_activations(1);
  const auto& local_output_grad = this->get_local_prev_error_signals();
  auto& local_values_grad = this->get_local_error_signals(0);
  auto& local_indices_grad = this->get_local_error_signals(1);
  
  const auto& input_dims_ = this->get_input_dims();
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> input_dims(input_dims_.begin(), input_dims_.end());
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());

  const size_t local_mini_batch_size = local_indices.Width();

  const size_t values_size = (input_dims.size()>1) ? input_dims[1] : this->get_input_size(0);
  const size_t output_size = (input_dims.size()>1) ?  this->get_output_dims()[1] : this->get_output_size();
  const size_t num_rows = (input_dims.size()>1) ? input_dims[0] : 1;
  // Zero out gradient w.r.t. indices
  El::Zero(local_indices_grad);

  // Gather into gradient w.r.t. values
  if (!local_values_grad.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_values_grad),
                                       gpu::get_sync_info(local_output_grad),
                                       gpu::get_sync_info(local_indices));
    constexpr size_t block_size = 16;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    block_dims.y = block_size;
    block_dims.z = 1;

    grid_dims.x = (num_rows + block_dims.y - 1) / block_dims.y;
    grid_dims.y = (values_size + block_dims.x - 1) / block_dims.x;
    grid_dims.z = (local_mini_batch_size + block_dims.z - 1) / block_dims.z;
    
    hydrogen::gpu::LaunchKernel(
      gather2d_kernel<TensorDataType>,
      grid_dims, block_dims, 0, multisync,
      local_indices.LockedBuffer(),
      Dim2{static_cast<size_t>(local_indices.LDim()), 1},
      local_output_grad.LockedBuffer(),
      Dim3{local_mini_batch_size, num_rows, output_size},
      Dim3{static_cast<size_t>(local_output_grad.LDim()), output_size, 1},
      local_values_grad.Buffer(),
      Dim3{local_mini_batch_size, num_rows, values_size},
      Dim3{static_cast<size_t>(local_values_grad.LDim()), num_rows, 1});
  }

}

#define PROTO(T)                                        \
  template class scatter_layer<                         \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
