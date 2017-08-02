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

// CUBLAS uses int for size parameters
template <typename ElmType>
cublasStatus_t Gemm(const cublasHandle_t &handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n, int k,
                    const ElmType alpha,
                    const ElmType *A, int lda,
                    const ElmType *B, int ldb,
                    const ElmType beta,
                    ElmType *C, int ldc);

template <> inline
cublasStatus_t Gemm<float>(const cublasHandle_t &handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m, int n, int k,
                           const float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float beta,
                           float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k,
                     &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <> inline
cublasStatus_t Gemm<double>(const cublasHandle_t &handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m, int n, int k,
                            const double alpha,
                            const double *A, int lda,
                            const double *B, int ldb,
                            const double beta,
                            double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k,
                     &alpha, A, lda, B, ldb, &beta, C, ldc);
}
                    
}
}

#endif // #ifdef __LIB_CUDA
#endif // CUBLAS_WRAPPER_HPP_INCLUDED
