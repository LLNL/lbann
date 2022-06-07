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

/** Copy between two device buffers, using all threads in a warp. */
#define LBANN_TRANSFORM_DISTCONV_NVSHMEM_VECTOR_ADDRESSING_INSTANTIATE
#include "lbann/utils/distconv.hpp"
#include "lbann/base.hpp"

#if defined(LBANN_HAS_NVSHMEM) && defined(LBANN_HAS_DISTCONV)

#include "lbann/layers/transform/distconv/distconv_nvshmem_vector_addressing.hpp"

namespace distconv{
  namespace{
  
  using Size2 = util::gpu_array<size_t, 2>;
  using Size3 = util::gpu_array<size_t, 3>;

  template <typename T> __device__ __forceinline__
  T* memcpy_warp(T* __restrict__ dest, const T* __restrict__ src, int n) {
    constexpr int warp_size = 32;
    for (int i = threadIdx.x; i < n; i += warp_size) {
      dest[i] = src[i];
    }
    __syncwarp();
    return dest;
  }

  /** Set device buffer, using all threads in a warp. */
  template <typename T> __device__ __forceinline__
  T* memset_warp(T* buf, T val, int n) {
    constexpr int warp_size = 32;
    for (int i = threadIdx.x; i < n; i += warp_size) {
      buf[i] = val;
    }
    __syncwarp();
    return buf;
  }

  /** See El::AbstractDistMatrix::ColOwner. */

  __device__ __forceinline__
  size_t distmat_index_owner(size_t global_index, size_t align, size_t stride) {
    // Figure out which rank owns currently holds this index
    // 
    return (global_index + align) % stride;
  }

  /** Get the location of the index in the global workspace*/ 
  __device__ __forceinline__
  size_t distmat_global_index(size_t local_index, size_t shift, size_t stride){
    // stride = len(index) 
    // shift = rank * stride
    return shift + (local_index * stride); 
  }

  /** Launch a collective NVSHMEM kernel.
 *
 *  Needed for device-side NVSHMEM synchronization calls like
 *  nvshmem_wait. If grid_dims is zero, then the NVSHMEM will launch
 *  with the largest available grid.
 *
 *  @todo Check that argument types match kernel signature.
 */
  template <typename Kernel, typename... Args>
  inline void launch_nvshmem_collective_kernel(
    const Kernel& kernel,
    dim3 grid_dims,
    dim3 block_dims,
    size_t shared_mem,
    cudaStream_t stream,
    Args... args) {
    if (grid_dims.x == 0) {
      grid_dims.y = 0;
      grid_dims.z = 0;
    }
    void* arg_list[] = {
      const_cast<void*>(reinterpret_cast<const void*>(&args))...
    };
    auto status = nvshmemx_collective_launch(
      reinterpret_cast<const void*>(&kernel),
      grid_dims,
      block_dims,
      arg_list,
      shared_mem,
      stream);
    if (status != 0) {
      LBANN_ERROR(
        "Failed to launch NVSHMEM collective kernel ",
        "(error ",status,")");
    }
  }

  /**
   * @brief Copy vector from shared memory heap to output tensor
   * 
   * Block dimensions: 32 x 1 x 1
   * 
   * Grid dimensions: input_dims[1] x input_dims[0] x 1
   * 
   */

  template <typename DataType>
  __global__ void copy_kernel(
    const DataType* __restrict__ src,
    DataType* __restrict__ dest,
    size_t local_mini_batch_size,
    size_t row_size, 
    size_t col_size ){
    
    const size_t bidx = blockIdx.x;
    const size_t bidy = blockIdx.y;
    const size_t nblocksx = gridDim.x;
    const size_t nblocksy = gridDim.y;

    const size_t i_per_block = (row_size + nblocksx -1) /nblocksx;
    const size_t i_start = bidx * i_per_block;
    size_t i_end = (bidx+1) * i_per_block;
    i_end = (i_end > row_size) ? i_end : row_size;

    for(size_t j = bidy; j < local_mini_batch_size; j+= nblocksy){
      for(size_t i = i_start; i < i_end; ++i){
        const auto& output_index = 0;
        const auto& workspace_index = 0; 
        memcpy_warp(&dest[output_index], &src[workspace_index], col_size);
      }
    } 
  }

  /**
   * @brief NVSHMEM Gather kernel
   * 
   * workspace(k, j, i) = values(k, indices(k, j), i)
   * 
   */

  template <typename DataType>
  __global__ void Gather_NVSHMEM_Kernel(
    const DataType* __restrict__ values,
    const Size3 values_shape,
    const DataType* __restrict__ indices,
    const Size2 indices_shape, 
    DataType* __restrict__ shared_buffer,
    const Size3 buffer_shape){

    const size_t bidx = blockIdx.x;
    const size_t bidy = blockIdx.y;
    const size_t gridDimx = gridDim.x;
    const size_t gridDimy = gridDim.y;
    const auto mini_batch_size = values_shape[0];
    const auto num_local_values_rows = values_shape[1];
    const auto num_local_values_cols = values_shape[2];
    
    // values_shape[0] == indices_shape[0]
    // For an initial implementation, assume that there is no data parallel decomposition
    // therefore, local_mini_batch_size = mini_batch_size 
    // So. buffer_shape[0] == values_shape[0]
    // - SZ
    
    const size_t i_per_block = (num_local_values_rows + gridDimx -1) / gridDimx;
    const size_t i_start = bidx * i_per_block;
    size_t i_end = (bidx+1) * i_per_block;
    i_end = (i_end > num_local_values_rows) ? i_end : num_local_values_rows;

    int n_pes = nvshmem_n_pes();
    
    for (size_t mb_i = bidy; mb_i < mini_batch_size; mb_i += gridDimy){
      for(size_t i = i_start; i < i_end; ++i){
        // Figure out which rank to send the vector
        const auto mb_offset = mb_i*mini_batch_size;
        const auto ind = static_cast<int>(std::floor(indices[mb_offset + i]));
        if (0<=ind && ind <static_cast<int>(num_local_values_rows)){
          
          const int pe = (ind - (ind % n_pes)) / n_pes;
          const int local_ind = ind % pe;
          
          nvshmemx_putmem_nbi_warp(&shared_buffer[mb_offset + local_ind * num_local_values_cols],
                                   &values[mb_offset + i * num_local_values_cols],
                                   num_local_values_cols * sizeof(DataType),
                                   pe);
        }
      }
    }
  }

  /**static_cast<DataType*>(m_buf.get())
   * @brief NVSHMEMScatter kernel
   * 
   */
  template <typename DataType>
  __global__ void Scatter_NVSHMEM_Kernel(
    const DataType* __restrict__ values,
    const Size3 values_shape,
    const DataType* __restrict__ indices,
    const Size2 indices_shape, 
    DataType* __restrict__ shared_buffer,
    const Size3 buffer_shape){

    const size_t bidx = blockIdx.x;
    const size_t bidy = blockIdx.y;
    const size_t gridDimx = gridDim.x;
    const size_t gridDimy = gridDim.y;
    const auto mini_batch_size = values_shape[0];
    const auto num_local_values_rows = values_shape[1];
    const auto num_local_values_cols = values_shape[2];

    const auto num_buffer_rows = buffer_shape[1];

    const size_t i_per_block = (num_local_values_rows + gridDimx -1) / gridDimx;
    const size_t i_start = bidx * i_per_block;
    size_t i_end = (bidx+1) * i_per_block;
    i_end = (i_end > num_local_values_rows) ? i_end : num_local_values_rows;

    int n_pes = nvshmem_n_pes();
    
    for (size_t mb_i = bidy; mb_i < mini_batch_size; mb_i += gridDimy){
      for(size_t i = i_start; i < i_end; ++i){
        // Figure out which rank to send the vector
        const auto mb_offset = mb_i*mini_batch_size;
        const auto ind = static_cast<int>(std::floor(indices[mb_offset + i]));
        if (0<=ind && ind <static_cast<int>(num_buffer_rows)){
          
          const int pe = (ind - (ind % n_pes)) / n_pes;
          const int local_ind = ind % pe;
          
          nvshmemx_getmem_warp(&shared_buffer[mb_offset + local_ind * num_local_values_cols],
                               &values[mb_offset + i * num_local_values_cols],
                               num_local_values_cols * sizeof(DataType),
                               pe);  // This is a blocking call 
        }
      }
    }
  } 
  } // namespace <anon>

  namespace tensor{
    
    template<typename DataType>
    void
    ScatterNVSHMEM<DataType>
    ::scatter(const DataType* values,
              const DataType* indices,
              DataType* output,
              const size_t local_mini_batch_size,
              const size_t values_rows_size,
              const size_t values_cols_size,
              const size_t output_rows_size){
                
      const auto buffer_size = local_mini_batch_size * values_rows_size * values_cols_size;
      const auto output_buffer_size = local_mini_batch_size * output_rows_size * values_cols_size;

      // Make sure the buffers are large enough
      ensure_buffer(buffer_size);
      ensure_output_buffer(output_buffer_size);

      constexpr size_t block_size = 32;
      dim3 block_dims, grid_dims;
      block_dims.x = block_size;
      grid_dims.x = output_rows_size;
      grid_dims.y = local_mini_batch_size;

      Size3 values_shape = {local_mini_batch_size, values_rows_size, values_cols_size};
      Size2 indices_shape = {local_mini_batch_size, values_rows_size};
      Size3 buffer_shape = {local_mini_batch_size, output_rows_size, values_cols_size};
      
      Scatter_NVSHMEM_Kernel<<<block_dims, grid_dims, 0, m_stream>>>(values,
                                                                      values_shape,
                                                                      indices,
                                                                      indices_shape,
                                                                      static_cast<DataType*>(m_buf.get()),
                                                                      buffer_shape);
      nvshmemx_quiet_on_stream(m_stream);
      copy_kernel<<<block_dims, grid_dims, 0, m_stream>>>(static_cast<DataType*>(m_output_buf.get()),
                                                          output,
                                                          local_mini_batch_size,
                                                          output_rows_size,
                                                          values_cols_size);
    }

    template<typename DataType>
    void
    GatherNVSHMEM<DataType>
    ::gather(const DataType* values,
             const DataType* indices,
             DataType* output,
             const size_t local_mini_batch_size,
             const size_t values_rows_size,
             const size_t values_cols_size,
             const size_t output_rows_size){

      const auto buffer_size = local_mini_batch_size * output_rows_size * values_cols_size;
      // Attach values matrix to the NVSHMEM buffer
      // The size of the NVSHMEM_values buffer is for the local values matrix
      // Retreive value vectors onto the NVSHMEM workspace buffer 
      // The NVSHMEM workspace buffer is the size of the local output matrix 
      // Copy the local workspace buffer onto the output matrix

      ensure_buffer(buffer_size);
      constexpr size_t block_size = 32;
      dim3 block_dims, grid_dims;
      block_dims.x = block_size;
      grid_dims.x = output_rows_size;
      grid_dims.y = local_mini_batch_size;

      Size3 values_shape = {local_mini_batch_size, values_rows_size, values_cols_size};
      Size2 indices_shape = {local_mini_batch_size, values_rows_size};
      Size3 buffer_shape = {local_mini_batch_size, output_rows_size, values_cols_size};
      Scatter_NVSHMEM_Kernel<<<block_dims, grid_dims, 0, m_stream>>>(values,
                                                                values_shape,
                                                                indices,
                                                                indices_shape,
                                                                static_cast<DataType*>(m_buf.get()),
                                                                buffer_shape);
      nvshmemx_quiet_on_stream(m_stream);
      copy_kernel<<<block_dims, grid_dims, 0, m_stream>>>(static_cast<DataType*>(m_buf.get()),
                                                          output,
                                                          local_mini_batch_size,
                                                          output_rows_size,
                                                          values_cols_size);
    }

  
  #define SCATTER_ETI(T)                            \
    template void ScatterNVSHMEM<T>::scatter(       \
      const T* values,                              \
      const T* indices,                             \
      T * output,                                   \
      const size_t local_mini_batch_size,           \
      const size_t values_rows_size,                \
      const size_t values_cols_size,                \
      const size_t output_rows_size);                    

  SCATTER_ETI(float)  
  SCATTER_ETI(double)
  #undef SCATTER_ETI

  #define GATHER_ETI(T)                            \
    template void GatherNVSHMEM<T>::gather(        \
      const T* values,                             \
      const T* indices,                            \
      T * output,                                  \
      const size_t local_mini_batch_size,          \
      const size_t values_rows_size,               \
      const size_t values_cols_size,               \
      const size_t output_rows_size);                    

  GATHER_ETI(float)  
  GATHER_ETI(double)
  #undef GATHER_ETI 

  } // namespace <distconv::tensor>
} //  namespace <distconv>
#endif 
