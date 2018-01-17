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
// cublass_wrapper .hpp - CUBLAS support - wrapper classes, utility functions
////////////////////////////////////////////////////////////////////////////////

#ifndef CUBLAS_WRAPPER_HPP_INCLUDED
#define CUBLAS_WRAPPER_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"

#ifdef __LIB_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace lbann {
namespace cublas {

// cuBLAS routines for floats
#if LBANN_DATATYPE == 4
inline
cublasStatus_t axpy(const cublasHandle_t &handle,
                    int n,
                    float alpha,
                    const float *x, int incx,
                    float *y, int incy) {
  return cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
}
inline
float nrm2(const cublasHandle_t &handle,
           int n, const float *x, int incx) {
  float result;
  CHECK_CUBLAS(cublasSnrm2(handle, n, x, incx, &result));
  return result;
}
inline
cublasStatus_t scal(const cublasHandle_t &handle,
                    int n,
                    float alpha,
                    float *x, int incx) {
  return cublasSscal(handle, n, &alpha, x, incx);
}
inline
cublasStatus_t gemv(const cublasHandle_t &handle,
                    cublasOperation_t trans,
                    int m, int n,
                    float alpha,
                    const float *A, int lda,
                    const float *x, int incx,
                    float beta,
                    float *y, int incy) {
  return cublasSgemv(handle, trans, m, n,
                     &alpha, A, lda, x, incx, &beta, y, incy);
}
inline
cublasStatus_t gemm(const cublasHandle_t &handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n, int k,
                    float alpha,
                    const float *A, int lda,
                    const float *B, int ldb,
                    float beta,
                    float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k,
                     &alpha, A, lda, B, ldb, &beta, C, ldc);
}
inline
cublasStatus_t geam(const cublasHandle_t &handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n,
                    float alpha, const float *A, int lda,
                    float beta, const float *B, int ldb,
                    float *C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n,
                     &alpha, A, lda, &beta, B, ldb, C, ldc);
}

// cuBLAS routines for doubles
#elif LBANN_DATATYPE == 8
inline
cublasStatus_t axpy(const cublasHandle_t &handle,
                    int n,
                    double alpha,
                    const double *x, int incx,
                    double *y, int incy) {
  return cublasDaxpy(handle, n, &alpha, x, incx, y, incy);
}
inline
double nrm2(const cublasHandle_t &handle,
            int n, const double *x, int incx) {
  double result;
  CHECK_CUBLAS(cublasDnrm2(handle, n, x, incx, &result));
  return result;
}
inline
cublasStatus_t scal(const cublasHandle_t &handle,
                    int n,
                    double alpha,
                    double *x, int incx) {
  return cublasDscal(handle, n, &alpha, x, incx);
}
inline
cublasStatus_t gemv(const cublasHandle_t &handle,
                    cublasOperation_t trans,
                    int m, int n,
                    double alpha,
                    const double *A, int lda,
                    const double *x, int incx,
                    double beta,
                    double *y, int incy) {
  return cublasDgemv(handle, trans, m, n,
                     &alpha, A, lda, x, incx, &beta, y, incy);
}
inline
cublasStatus_t gemm(const cublasHandle_t &handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n, int k,
                    double alpha,
                    const double *A, int lda,
                    const double *B, int ldb,
                    double beta,
                    double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k,
                     &alpha, A, lda, B, ldb, &beta, C, ldc);
}
inline
cublasStatus_t geam(const cublasHandle_t &handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n,
                    double alpha, const double *A, int lda,
                    double beta, const double *B, int ldb,
                    double *C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n,
                     &alpha, A, lda, &beta, B, ldb, C, ldc);
}

// cuBLAS routines for an invalid datatype
#else
#error Invalid floating-point datatype (must be float or double)
#endif
                    
}
}

#endif // #ifdef __LIB_CUDA
#endif // CUBLAS_WRAPPER_HPP_INCLUDED
