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

#ifdef LBANN_HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace lbann {
namespace cublas {
namespace internal {

template <typename T>
struct cuBLAS_Caller;

template <>
struct cuBLAS_Caller<float>
{
    template <typename... Ts>
    cublasStatus_t axpy(Ts&&... args)
    {
        return cublasSaxpy(std::forward<Ts>(args)...);
    }

    template <typename... Ts>
    float nrm2(Ts&&... args)
    {
        float result;
        CHECK_CUBLAS(cublasSnrm2(std::forward<Ts>(args)..., &result));
        return result;
    }

    template <typename... Ts>
    cublasStatus_t scal(Ts&&... args)
    {
        return cublasSscal(std::forward<Ts>(args)...);
    }

    template <typename... Ts>
    cublasStatus_t gemm(Ts&&... args)
    {
        return cublasSgemm(std::forward<Ts>(args)...);
    }
};

template <>
struct cuBLAS_Caller<double>
{
    template <typename... Ts>
    cublasStatus_t axpy(Ts&&... args)
    {
        return cublasDaxpy(std::forward<Ts>(args)...);
    }

    template <typename... Ts>
    double nrm2(Ts&&... args)
    {
        double result;
        CHECK_CUBLAS(cublasDnrm2(std::forward<Ts>(args)..., &result));
        return result;
    }

    template <typename... Ts>
    cublasStatus_t scal(Ts&&... args)
    {
        return cublasDscal(std::forward<Ts>(args)...);
    }

    template <typename... Ts>
    cublasStatus_t gemm(Ts&&... args)
    {
        return cublasDgemm(std::forward<Ts>(args)...);
    }
};

}// namespace internal

inline cublasStatus_t axpy(cublasHandle_t const& handle,
                           int n, DataType alpha, DataType const* x, int incx,
                           DataType * y, int incy)
{
    return internal::cuBLAS_Caller<DataType>{}.axpy(
        handle, n, &alpha, x, incx, y, incy);
}

inline DataType nrm2(cublasHandle_t const& handle,
                     int n, DataType const* x, int incx)
{
    return internal::cuBLAS_Caller<DataType>{}.nrm2(handle, n, x, incx);
}

inline cublasStatus_t scal(cublasHandle_t const& handle,
                           int n, DataType alpha, DataType * x, int incx)
{
    return internal::cuBLAS_Caller<DataType>{}.scal(handle, n, &alpha, x, incx);
}

inline cublasStatus_t gemm(cublasHandle_t const& handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           DataType alpha,
                           DataType const * A, int lda,
                           DataType const * B, int ldb,
                           DataType beta,
                           DataType * C, int ldc)
{
    return internal::cuBLAS_Caller<DataType>{}.gemm(
        handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

}// namespace cublas
}// namespace lbann

#endif // #ifdef LBANN_HAS_CUDA
#endif // CUBLAS_WRAPPER_HPP_INCLUDED
