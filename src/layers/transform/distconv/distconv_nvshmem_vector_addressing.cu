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
#include "lbann/layers/transform/distconv/distconv_scatter.hpp"
namespace distconv{
  namespace{

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
        "(error ",status,")");s
  }

  } // anonymous namespace <anon>

  namespace tensor{
    
  } // namespace <distconv::tensor>
} //  namespace <distconv>