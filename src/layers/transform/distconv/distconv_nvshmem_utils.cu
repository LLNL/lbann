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

