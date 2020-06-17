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

#include <lbann_config.hpp>
#include "lbann/utils/cublas.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_CUDA

// Wrapper function macro
#define WRAP_CUBLAS(function, wrapper)                          \
  template <typename... Ts>                                     \
  void wrapper(Ts&&... args) {                                  \
    CHECK_CUBLAS(function(std::forward<Ts>(args)...));          \
  }

#define ERROR_OUT(wrapper)                                          \
  template <typename... Ts>                                         \
  void wrapper(Ts&&... args) {                                      \
    LBANN_ERROR("Cannot dispatch " #wrapper "() for this type.");   \
  }

namespace {

template <typename T>
struct cuBLAS_Caller;

template <>
struct cuBLAS_Caller<float> {
  WRAP_CUBLAS(cublasSaxpy, axpy)
  WRAP_CUBLAS(cublasSdot, dot)
  WRAP_CUBLAS(cublasSnrm2, nrm2)
  WRAP_CUBLAS(cublasSscal, scal)
  WRAP_CUBLAS(cublasSgemv, gemv)
  WRAP_CUBLAS(cublasSgemm, gemm)
  WRAP_CUBLAS(cublasSgeam, geam)
  WRAP_CUBLAS(cublasSgemmStridedBatched, gemm_strided_batched)
};

template <>
struct cuBLAS_Caller<double> {
  WRAP_CUBLAS(cublasDaxpy, axpy)
  WRAP_CUBLAS(cublasDdot, dot)
  WRAP_CUBLAS(cublasDnrm2, nrm2)
  WRAP_CUBLAS(cublasDscal, scal)
  WRAP_CUBLAS(cublasDgemv, gemv)
  WRAP_CUBLAS(cublasDgemm, gemm)
  WRAP_CUBLAS(cublasDgeam, geam)
  WRAP_CUBLAS(cublasDgemmStridedBatched, gemm_strided_batched)
};

#ifdef LBANN_HAS_GPU_FP16
template <>
struct cuBLAS_Caller<__half> {
  void axpy(cublasHandle_t handle, int n,
            __half const* alpha,
            __half const* x, int incx,
            __half* y, int incy)
  {
    CHECK_CUBLAS(
      cublasAxpyEx(handle, n, alpha, CUDA_R_16F, x, CUDA_R_16F, incx,
                   y, CUDA_R_16F, incy, CUDA_R_32F));
  }

  void dot(cublasHandle_t handle, int n,
           __half const* x, int incx,
           __half const* y, int incy,
           __half* result)
  {
    CHECK_CUBLAS(
      cublasDotEx(handle, n, x, CUDA_R_16F, incx, y, CUDA_R_16F, incy,
                  result, CUDA_R_16F,  CUDA_R_32F));
  }

  void nrm2(cublasHandle_t handle, int n, __half const* x, int incx,
            __half* result)
  {
    CHECK_CUBLAS(
      cublasNrm2Ex(handle, n, x, CUDA_R_16F, incx,
                   result, CUDA_R_16F, CUDA_R_32F));
  }

  void scal(cublasHandle_t handle, int n,
            __half const* alpha,
            __half* x, int incx)
  {
    CHECK_CUBLAS(
      cublasScalEx(handle, n, alpha, CUDA_R_16F,
                   x, CUDA_R_16F, incx,
                   CUDA_R_32F));
  }

  WRAP_CUBLAS(cublasHgemm, gemm)
  WRAP_CUBLAS(cublasHgemmStridedBatched, gemm_strided_batched)

  ERROR_OUT(geam)
  ERROR_OUT(gemv)
};
#endif // LBANN_HAS_GPU_FP16
} // namespace

namespace lbann {
namespace cublas {

const std::string get_error_string(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "unknown cuBLAS error";
  }
}

template <typename TensorDataType>
void axpy(cublasHandle_t const& handle,
          int n,
          TensorDataType alpha,
          TensorDataType const* x, int incx,
          TensorDataType * y, int incy) {
  cuBLAS_Caller<TensorDataType>{}.axpy(handle, n, &alpha, x, incx, y, incy);
}

template <typename TensorDataType>
void dot(cublasHandle_t const& handle,
         int n,
         TensorDataType const* x, int incx,
         TensorDataType const* y, int incy,
         TensorDataType * result) {
  cuBLAS_Caller<TensorDataType>{}.dot(handle, n, x, incx, y, incy, result);
}

template <typename TensorDataType>
TensorDataType dot(cublasHandle_t const& handle,
             int n,
             TensorDataType const* x, int incx,
             TensorDataType const* y, int incy) {
  TensorDataType result;
  dot(handle, n, x, incx, y, incy, &result);
  return result;
}

template <typename TensorDataType>
void nrm2(cublasHandle_t const& handle,
          int n,
          TensorDataType const* x, int incx,
          TensorDataType * result) {
  cuBLAS_Caller<TensorDataType>{}.nrm2(handle, n, x, incx, result);
}

template <typename TensorDataType>
TensorDataType nrm2(cublasHandle_t const& handle,
              int n,
              TensorDataType const* x, int incx) {
  TensorDataType result;
  nrm2(handle, n, x, incx, &result);
  return result;

}

template <typename TensorDataType>
void scal(cublasHandle_t const& handle,
          int n,
          TensorDataType alpha,
          TensorDataType * x, int incx) {
  cuBLAS_Caller<TensorDataType>{}.scal(handle, n, &alpha, x, incx);
}

template <typename TensorDataType>
void gemv(cublasHandle_t const& handle,
          cublasOperation_t trans,
          int m, int n,
          TensorDataType alpha,
          TensorDataType const * A, int lda,
          TensorDataType const * x, int incx,
          TensorDataType beta,
          TensorDataType * y, int incy) {
  cuBLAS_Caller<TensorDataType>{}.gemv(handle, trans, m, n,
                                 &alpha, A, lda, x, incx, &beta, y, incy);
}

template <typename TensorDataType>
void gemm(cublasHandle_t const& handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          TensorDataType alpha,
          TensorDataType const * A, int lda,
          TensorDataType const * B, int ldb,
          TensorDataType beta,
          TensorDataType * C, int ldc) {
  cuBLAS_Caller<TensorDataType>{}.gemm(handle, transa, transb, m, n, k,
                                 &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <typename TensorDataType>
void geam(cublasHandle_t const& handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n,
          TensorDataType alpha,
          TensorDataType const * A, int lda,
          TensorDataType beta,
          TensorDataType const * B, int ldb,
          TensorDataType * C, int ldc) {
  cuBLAS_Caller<TensorDataType>{}.geam(handle, transa, transb, m, n,
                                 &alpha, A, lda, &beta, B, ldb, C, ldc);
}

template <typename TensorDataType>
void gemm_strided_batched(cublasHandle_t const& handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          TensorDataType alpha,
                          TensorDataType const * A, int lda,
                          long long int strideA,
                          TensorDataType const * B, int ldb,
                          long long int strideB,
                          TensorDataType beta,
                          TensorDataType * C, int ldc,
                          long long int strideC,
                          int batchCount) {
  cuBLAS_Caller<TensorDataType>{}.gemm_strided_batched(
    handle, transa, transb, m, n, k,
    &alpha, A, lda, strideA, B, ldb, strideB,
    &beta, C, ldc, strideC, batchCount);
}

#define PROTO(T)                                                                       \
  template void axpy<T>(cublasHandle_t const&, int, T, T const*, int, T *, int);       \
  template void dot<T>(cublasHandle_t const&, int, T const*, int, T const*, int, T *); \
  template T dot<T>(cublasHandle_t const&, int, T const*, int, T const*, int);         \
  template void nrm2<T>(cublasHandle_t const&, int, T const*, int, T *);               \
  template T nrm2<T>(cublasHandle_t const&, int, T const*, int);                       \
  template void scal<T>(cublasHandle_t const&, int, T, T *, int);                      \
  template void gemv<T>(cublasHandle_t const&, cublasOperation_t, int, int, T,         \
    T const *, int, T const *, int, T, T *, int);                                      \
  template void gemm<T>(cublasHandle_t const&, cublasOperation_t, cublasOperation_t,   \
    int, int, int, T, T const *, int, T const *, int, T, T *, int);                    \
  template void geam<T>(cublasHandle_t const&, cublasOperation_t, cublasOperation_t,   \
    int, int, T, T const *, int, T, T const *, int, T *, int);                         \
  template void gemm_strided_batched<T>(cublasHandle_t const&, cublasOperation_t,      \
    cublasOperation_t, int, int, int, T, T const *, int, long long int, T const *,     \
    int, long long int, T, T *, int, long long int, int)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

void default_to_tensor_ops()
{
  CHECK_CUBLAS(
        cublasSetMathMode(
          hydrogen::cublas::GetLibraryHandle(),
          CUBLAS_TENSOR_OP_MATH));
}

} // namespace cublas
} // namespace lbann

#endif // LBANN_HAS_CUDA
