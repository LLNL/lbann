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
#ifndef LBANN_SRC_LAYERS_TRANSFORM_CUTENSOR_SUPPORT_HPP_INCLUDED
#define LBANN_SRC_LAYERS_TRANSFORM_CUTENSOR_SUPPORT_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"

#include <cuda_runtime.h>
#include <cutensor.h>

#define CHECK_CUTENSOR(cmd)                                                    \
  do {                                                                         \
    auto const lbann_chk_cutensor_status__ = (cmd);                            \
    if (CUTENSOR_STATUS_SUCCESS != lbann_chk_cutensor_status__) {              \
      LBANN_ERROR("cuTENSOR error (status=",                                   \
                  lbann_chk_cutensor_status__,                                 \
                  "): ",                                                       \
                  cutensorGetErrorString(lbann_chk_cutensor_status__));        \
    }                                                                          \
  } while (false)

namespace lbann {

template <typename CppType>
struct CUDATypeT;

template <>
struct CUDATypeT<__half>
{
  typedef float scalar_type;
  static constexpr auto value = CUDA_R_16F;
};

template <>
struct CUDATypeT<float>
{
  typedef float scalar_type;
  static constexpr auto value = CUDA_R_32F;
};
template <>
struct CUDATypeT<double>
{
  typedef double scalar_type;
  static constexpr auto value = CUDA_R_64F;
};
template <>
struct CUDATypeT<El::Complex<float>>
{
  typedef El::Complex<float> scalar_type;
  static constexpr auto value = CUDA_C_32F;
};
template <>
struct CUDATypeT<El::Complex<double>>
{
  typedef El::Complex<double> scalar_type;
  static constexpr auto value = CUDA_C_64F;
};

template <typename CppType>
constexpr auto CUDAType = CUDATypeT<CppType>::value;

template <typename CppType>
using CUDAScalar = typename CUDATypeT<CppType>::scalar_type;

template <typename CppType>
constexpr auto CUDAScalarType = CUDATypeT<CUDAScalar<CppType>>::value;

static cutensorHandle_t make_handle()
{
  cutensorHandle_t handle;
  CHECK_CUTENSOR(cutensorInit(&handle));
  return handle;
}
static cutensorHandle_t* get_handle_ptr()
{
  static cutensorHandle_t handle = make_handle();
  return &handle;
}

} // namespace lbann

#endif // LBANN_SRC_LAYERS_TRANSFORM_CUTENSOR_SUPPORT_HPP_INCLUDED
