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

#include "lbann/utils/cublas.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_CUDA

// Wrapper function macro
#define WRAP_CUBLAS(function, wrapper)                          \
  template <typename... Ts>                                     \
  void wrapper(Ts&&... args) {                                  \
    CHECK_CUBLAS(function(std::forward<Ts>(args)...));          \
  }
namespace {

template <typename T>
struct cuBLAS_Caller;

template <>
struct cuBLAS_Caller<float> {
  WRAP_CUBLAS(cublasSaxpy, axpy)
  WRAP_CUBLAS(cublasSdot , dot )
  WRAP_CUBLAS(cublasSnrm2, nrm2)
  WRAP_CUBLAS(cublasSscal, scal)
  WRAP_CUBLAS(cublasSgemv, gemv)
  WRAP_CUBLAS(cublasSgemm, gemm)
  WRAP_CUBLAS(cublasSgeam, geam)
};

template <>
struct cuBLAS_Caller<double> {
  WRAP_CUBLAS(cublasDaxpy, axpy)
  WRAP_CUBLAS(cublasDdot , dot )
  WRAP_CUBLAS(cublasDnrm2, nrm2)
  WRAP_CUBLAS(cublasDscal, scal)
  WRAP_CUBLAS(cublasDgemv, gemv)
  WRAP_CUBLAS(cublasDgemm, gemm)
  WRAP_CUBLAS(cublasDgeam, geam)
};

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

void axpy(cublasHandle_t const& handle,
          int n,
          DataType alpha,
          DataType const* x, int incx,
          DataType * y, int incy) {
  cuBLAS_Caller<DataType>{}.axpy(handle, n, &alpha, x, incx, y, incy);
}

void dot(cublasHandle_t const& handle,
         int n,
         DataType const* x, int incx,
         DataType const* y, int incy,
         DataType * result) {
  cuBLAS_Caller<DataType>{}.dot(handle, n, x, incx, y, incy, result);
}

DataType dot(cublasHandle_t const& handle,
             int n,
             DataType const* x, int incx,
             DataType const* y, int incy) {
  DataType result;
  dot(handle, n, x, incx, y, incy, &result);
  return result;
}

void nrm2(cublasHandle_t const& handle,
          int n,
          DataType const* x, int incx,
          DataType * result) {
  cuBLAS_Caller<DataType>{}.nrm2(handle, n, x, incx, result);
}

DataType nrm2(cublasHandle_t const& handle,
              int n,
              DataType const* x, int incx) {
  DataType result;
  nrm2(handle, n, x, incx, &result);
  return result;

}

void scal(cublasHandle_t const& handle,
          int n,
          DataType alpha,
          DataType * x, int incx) {
  cuBLAS_Caller<DataType>{}.scal(handle, n, &alpha, x, incx);
}

void gemv(cublasHandle_t const& handle,
          cublasOperation_t trans,
          int m, int n,
          DataType alpha,
          DataType const * A, int lda,
          DataType const * x, int incx,
          DataType beta,
          DataType * y, int incy) {
  cuBLAS_Caller<DataType>{}.gemv(handle, trans, m, n,
                                 &alpha, A, lda, x, incx, &beta, y, incy);
}

void gemm(cublasHandle_t const& handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          DataType alpha,
          DataType const * A, int lda,
          DataType const * B, int ldb,
          DataType beta,
          DataType * C, int ldc) {
  cuBLAS_Caller<DataType>{}.gemm(handle, transa, transb, m, n, k,
                                 &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void geam(cublasHandle_t const& handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n,
          DataType alpha,
          DataType const * A, int lda,
          DataType beta,
          DataType const * B, int ldb,
          DataType * C, int ldc) {
  cuBLAS_Caller<DataType>{}.geam(handle, transa, transb, m, n,
                                 &alpha, A, lda, &beta, B, ldb, C, ldc);
}

} // namespace cublas
} // namespace lbann

#endif // LBANN_HAS_CUDA
