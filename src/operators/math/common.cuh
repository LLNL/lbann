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
#ifndef LBANN_SRC_OPERATORS_MATH_COMMON_CUH_INCLUDED
#define LBANN_SRC_OPERATORS_MATH_COMMON_CUH_INCLUDED

#if defined __CUDACC__ || defined __HIPCC__

#include "lbann/base.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {
namespace internal {
namespace kernel {

/** @brief Apply a functor to a 2-D column-major matrix buffer.
 *
 *  This can be applied to a row-major matrix by logically transposing
 *  the matrix.
 *
 *  @tparam TILE_DIM The number of rows/columns being processed by a
 *                   thread block.
 *  @tparam BLK_COLS The number of columns handled at one time in the
 *                   block.
 *
 *  @tparam S (Inferred) Type of first source buffer.
 *  @tparam T (Inferred) Type of second source buffer.
 *  @tparam U (Inferred) Type of target buffer.
 *  @tparam SizeT (Inferred) Type of integer used to express sizes.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `T(S const&)`.
 *
 *  @param m The number of rows in A/B/C.
 *  @param n The number of columns in A/B/C. Columns must be contiguous
 *           in memory.
 *  @param A The first source matrix buffer.
 *  @param lda The stride between columns of A in terms of elements of
 *             type S.
 *  @param B The second source matrix buffer.
 *  @param ldb The stride between columns of B in terms of elements of
 *             type T.
 *  @param C The target matrix buffer.
 *  @param ldc The stride between columns of C in terms of elements of
 *             type U.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <int TILE_DIM,
          int BLK_COLS,
          typename S,
          typename T,
          typename U,
          typename SizeT,
          typename FunctorT>
__global__ void entrywise_zip_into_kernel_naive(
    SizeT m, SizeT n,
    S const* const __restrict__  A, SizeT lda,
    T const* const __restrict__  B, SizeT ldb,
    U * const __restrict__ C, SizeT ldc,
    FunctorT func)
{
    size_t const row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t const col_idx = blockIdx.y * TILE_DIM + threadIdx.y;

    if (row_idx < m)
    {
        for (int ii = 0; ii < TILE_DIM && col_idx + ii < n; ii += BLK_COLS)
            C[row_idx + (col_idx+ii)*ldc] =
                func(A[row_idx + (col_idx+ii)*lda],
                     B[row_idx + (col_idx+ii)*ldb]);
    }
}

    /** CUDA kernel to apply an binary backprop operator. */
template <typename DataT, typename F>
__global__
void binary_backprop_operator_kernel(El::Int height, El::Int width,
                                     DataT const* const __restrict__ x1,
                                     El::Int x1_ldim,
                                     DataT const* const __restrict__ x2,
                                     El::Int x2_ldim,
                                     DataT const* const __restrict__ dy,
                                     El::Int dy_ldim,
                                     DataT* const __restrict__ dx1,
                                     El::Int dx1_ldim,
                                     DataT* const __restrict__ dx2,
                                     El::Int dx2_ldim,
                                     F func)
{
  El::Int const gid = threadIdx.x + blockIdx.x * blockDim.x;
  El::Int const size = height * width;
  El::Int const num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    auto const& row = pos % height;
    auto const& col = pos / height;
    func(x1[row + col * x1_ldim],
         x2[row + col * x2_ldim],
         dy[row + col * dy_ldim],
         dx1[row + col * dx1_ldim],
         dx2[row + col * dx2_ldim]);
  }
}

/** @brief Apply a functor to a 2-D column-major matrix buffer.
 *
 *  This can be applied to a row-major matrix by logically transposing
 *  the matrix.
 *
 *  @tparam TILE_DIM The number of rows/columns being processed by a
 *                   thread block.
 *  @tparam BLK_COLS The number of columns handled at one time in the
 *                   block.
 *
 *  @tparam S (Inferred) Type of first source buffer.
 *  @tparam T (Inferred) Type of second source buffer.
 *  @tparam U (Inferred) Type of third source buffer.
 *  @tparam R (Inferred) Type of target buffer.
 *  @tparam SizeT (Inferred) Type of integer used to express sizes.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `R(S const&, T const&, U const&)`.
 *
 *  @param m The number of rows in A/B/C.
 *  @param n The number of columns in A/B/C. Columns must be contiguous
 *           in memory.
 *  @param A The first source matrix buffer.
 *  @param lda The stride between columns of A in terms of elements of
 *             type S.
 *  @param B The second source matrix buffer.
 *  @param ldb The stride between columns of B in terms of elements of
 *             type T.
 *  @param C The third source matrix buffer.
 *  @param ldc The stride between columns of C in terms of elements of
 *             type U.
 *  @param D The target matrix buffer.
 *  @param ldd The stride between columns of D in terms of elements of
 *             type R.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <int TILE_DIM,
          int BLK_COLS,
          typename S,
          typename T,
          typename U,
          typename R,
          typename SizeT,
          typename FunctorT>
__global__ void entrywise_zip_into_kernel_naive(
    SizeT m, SizeT n,
    S const* const __restrict__  A, SizeT lda,
    T const* const __restrict__  B, SizeT ldb,
    U const* const __restrict__  C, SizeT ldc,
    R * const __restrict__ D, SizeT ldd,
    FunctorT func)
{
    size_t const row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t const col_idx = blockIdx.y * TILE_DIM + threadIdx.y;

    if (row_idx < m)
    {
        for (int ii = 0; ii < TILE_DIM && col_idx + ii < n; ii += BLK_COLS)
            D[row_idx + (col_idx+ii)*ldd] =
                func(A[row_idx + (col_idx+ii)*lda],
                     B[row_idx + (col_idx+ii)*ldb],
                     C[row_idx + (col_idx+ii)*ldc]);
    }
}

/** CUDA kernel to apply a ternary backprop operator. */
template <typename DataT, typename F>
__global__
void ternary_backprop_operator_kernel(El::Int height, El::Int width,
                                      DataT const* const __restrict__ x1,
                                      El::Int x1_ldim,
                                      DataT const* const __restrict__ x2,
                                      El::Int x2_ldim,
                                      DataT const* const __restrict__ x3,
                                      El::Int x3_ldim,
                                      DataT const* const __restrict__ dy,
                                      El::Int dy_ldim,
                                      DataT* const __restrict__ dx1,
                                      El::Int dx1_ldim,
                                      DataT* const __restrict__ dx2,
                                      El::Int dx2_ldim,
                                      DataT* const __restrict__ dx3,
                                      El::Int dx3_ldim,
                                      F func)
{
  El::Int const gid = threadIdx.x + blockIdx.x * blockDim.x;
  El::Int const size = height * width;
  El::Int const num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    auto const& row = pos % height;
    auto const& col = pos / height;
    func(x1[row + col * x1_ldim],
         x2[row + col * x2_ldim],
         x3[row + col * x3_ldim],
         dy[row + col * dy_ldim],
         dx1[row + col * dx1_ldim],
         dx2[row + col * dx2_ldim],
         dx3[row + col * dx3_ldim]);
  }
}

}// namespace kernel

/** @brief Apply a functor to 2-D column-major matrix buffers.
 *
 *  @warning Calling this function is only valid in device-compiled code.
 *
 *  @tparam S (Inferred) Type of first source buffer.
 *  @tparam T (Inferred) Type of second source buffer.
 *  @tparam U (Inferred) Type of target buffer.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `U(S const&, T const&)`.
 *
 *  @param A The first source matrix.
 *  @param B The second source matrix.
 *  @param B The target matrix.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <typename S, typename T, typename U, typename FunctorT>
void EntrywiseZipInto(El::Matrix<S, El::Device::GPU> const& A,
                      El::Matrix<T, El::Device::GPU> const& B,
                      El::Matrix<U, El::Device::GPU>& C,
                      FunctorT func)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(C),
                                     gpu::get_sync_info(A),
                                     gpu::get_sync_info(B));

  auto const m = A.Height();
  auto const n = A.Width();
  if (m == El::TypeTraits<El::Int>::Zero()
      || n == El::TypeTraits<El::Int>::Zero())
  {
    // Nothing to do
    return;
  }

  constexpr int TILE_DIM = El::gpu::Default2DTileSize();
  constexpr int BLK_COLS = 8;

  static_assert(TILE_DIM % BLK_COLS == 0,
                "Incompatible TILE_DIM, BLK_COLS.");

  dim3 blks((m + TILE_DIM - 1) / TILE_DIM,
            (n + TILE_DIM - 1) / TILE_DIM,
            1);
  dim3 thds(TILE_DIM, BLK_COLS, 1);

  El::gpu::LaunchKernel(
    kernel::entrywise_zip_into_kernel_naive<
      TILE_DIM,
      BLK_COLS,
      El::NativeGPUType<S>,
      El::NativeGPUType<T>,
      El::NativeGPUType<U>,
      El::Int,
      FunctorT>,
    blks, thds, 0, multisync,
    m, n,
    El::AsNativeGPUType(A.LockedBuffer()), A.LDim(),
    El::AsNativeGPUType(B.LockedBuffer()), B.LDim(),
    El::AsNativeGPUType(C.Buffer()), C.LDim(),
    func);
}


/** Apply a binary backprop operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions. Given a binary function \f$ y = f(x_1,x_2) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$. The last two arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <typename DataT, typename F>
void apply_binary_backprop_operator(El::Matrix<DataT, El::Device::GPU> const& x1,
                                    El::Matrix<DataT, El::Device::GPU> const& x2,
                                    El::Matrix<DataT, El::Device::GPU> const& dy,
                                    El::Matrix<DataT, El::Device::GPU>& dx1,
                                    El::Matrix<DataT, El::Device::GPU>& dx2,
                                    F func)
{
  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  El::Int const height = x1.Height();
  El::Int const width = x1.Width();
  El::Int const block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx2),
                                       gpu::get_sync_info(dx1),
                                       gpu::get_sync_info(dy),
                                       gpu::get_sync_info(x2),
                                       gpu::get_sync_info(x1));
    hydrogen::gpu::LaunchKernel(
      kernel::binary_backprop_operator_kernel<DataT, F>,
      grid_dim, block_dim, 0, multisync,
      height, width,
      x1.LockedBuffer(), x1.LDim(),
      x2.LockedBuffer(), x2.LDim(),
      dy.LockedBuffer(), dy.LDim(),
      dx1.Buffer(), dx1.LDim(),
      dx2.Buffer(), dx2.LDim(),
      func);
  }
}

/** @brief Apply a functor to 2-D column-major matrix buffers.
 *
 *  @warning Calling this function is only valid in device-compiled code.
 *
 *  @tparam S (Inferred) Type of first source buffer.
 *  @tparam T (Inferred) Type of second source buffer.
 *  @tparam U (Inferred) Type of third source buffer.
 *  @tparam R (Inferred) Type of target buffer.
 *  @tparam FunctorT (Inferred) Type of functor. Must be equivalent to
 *                              `R(S const&, T const&, U const&)`.
 *
 *  @param A The first source matrix.
 *  @param B The second source matrix.
 *  @param C The third source matrix.
 *  @param D The target matrix.
 *  @param func The functor to apply. Must be device-invocable.
 */
template <typename S, typename T, typename U, typename R, typename FunctorT>
void EntrywiseZipInto(El::Matrix<S, El::Device::GPU> const& A,
                      El::Matrix<T, El::Device::GPU> const& B,
                      El::Matrix<U, El::Device::GPU> const& C,
                      El::Matrix<R, El::Device::GPU>& D,
                      FunctorT func)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(D),
                                     gpu::get_sync_info(A),
                                     gpu::get_sync_info(B),
                                     gpu::get_sync_info(C));
  auto const m = A.Height();
  auto const n = A.Width();
  if (m == El::TypeTraits<El::Int>::Zero()
      || n == El::TypeTraits<El::Int>::Zero())
  {
    // Nothing to do
    return;
  }

  constexpr int TILE_DIM = El::gpu::Default2DTileSize();
  constexpr int BLK_COLS = 8;

  static_assert(TILE_DIM % BLK_COLS == 0,
                "Incompatible TILE_DIM, BLK_COLS.");

  dim3 blks((m + TILE_DIM - 1) / TILE_DIM,
            (n + TILE_DIM - 1) / TILE_DIM,
            1);
  dim3 thds(TILE_DIM, BLK_COLS, 1);

  El::gpu::LaunchKernel(
    kernel::entrywise_zip_into_kernel_naive<
      TILE_DIM,
      BLK_COLS,
      El::NativeGPUType<S>,
      El::NativeGPUType<T>,
      El::NativeGPUType<U>,
      El::NativeGPUType<R>,
      El::Int,
      FunctorT>,
    blks, thds, 0, multisync,
    m, n,
    El::AsNativeGPUType(A.LockedBuffer()), A.LDim(),
    El::AsNativeGPUType(B.LockedBuffer()), B.LDim(),
    El::AsNativeGPUType(C.LockedBuffer()), C.LDim(),
    El::AsNativeGPUType(D.Buffer()), D.LDim(),
    func);
}


/** Apply a ternary backprop operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions. Given a ternary function \f$ y = f(x_1,x_2,x_3) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ x_3 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$, \f$ dL/dx_3 \f$. The last three arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <typename DataT, typename F>
void apply_ternary_backprop_operator(El::Matrix<DataT, El::Device::GPU> const& x1,
                                    El::Matrix<DataT, El::Device::GPU> const& x2,
                                    El::Matrix<DataT, El::Device::GPU> const& x3,
                                    El::Matrix<DataT, El::Device::GPU> const& dy,
                                    El::Matrix<DataT, El::Device::GPU>& dx1,
                                    El::Matrix<DataT, El::Device::GPU>& dx2,
                                    El::Matrix<DataT, El::Device::GPU>& dx3,
                                    F func)
{
  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  El::Int const height = x1.Height();
  El::Int const width = x1.Width();
  El::Int const block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx3),
                                       gpu::get_sync_info(dx2),
                                       gpu::get_sync_info(dx1),
                                       gpu::get_sync_info(dy),
                                       gpu::get_sync_info(x3),
                                       gpu::get_sync_info(x2),
                                       gpu::get_sync_info(x1));
    hydrogen::gpu::LaunchKernel(
      kernel::ternary_backprop_operator_kernel<DataT, F>,
      grid_dim, block_dim, 0, multisync,
      height, width,
      x1.LockedBuffer(), x1.LDim(),
      x2.LockedBuffer(), x2.LDim(),
      x3.LockedBuffer(), x3.LDim(),
      dy.LockedBuffer(), dy.LDim(),
      dx1.Buffer(), dx1.LDim(),
      dx2.Buffer(), dx2.LDim(),
      dx3.Buffer(), dx3.LDim(),
      func);
  }
}

}  // namespace internal
}  // namespace lbann
#endif // defined __CUDACC__ || defined __HIPCC__
#endif // LBANN_SRC_OPERATORS_MATH_COMMON_CUH_INCLUDED
