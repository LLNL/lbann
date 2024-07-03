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
#ifndef LBANN_UTILS_CUTENSOR_SUPPORT_HPP_INCLUDED
#define LBANN_UTILS_CUTENSOR_SUPPORT_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/tensor_dims_utils.hpp"
#include "lbann/utils/typename.hpp"

#include <cuda_runtime.h>
#include <cutensor.h>

/**
 * The interface below is designed for CUTENSOR v1.
 **/

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

namespace cutensor {
using DimsType = ColMajorDims<int64_t>;
using StridesType = ColMajorStrides<int64_t>;
using ModesType = std::vector<int32_t>;
} // namespace cutensor

template <typename CppType>
struct CUDATypeT;

template <>
struct CUDATypeT<__half>
{
  typedef float scalar_type;
  static constexpr auto value = CUDA_R_16F;
  static constexpr auto compute_type = CUTENSOR_COMPUTE_16F;
};

template <>
struct CUDATypeT<float>
{
  typedef float scalar_type;
  static constexpr auto value = CUDA_R_32F;
  static constexpr auto compute_type = CUTENSOR_COMPUTE_32F;
};
template <>
struct CUDATypeT<double>
{
  typedef double scalar_type;
  static constexpr auto value = CUDA_R_64F;
  static constexpr auto compute_type = CUTENSOR_COMPUTE_64F;
};
template <>
struct CUDATypeT<El::Complex<float>>
{
  typedef El::Complex<float> scalar_type;
  static constexpr auto value = CUDA_C_32F;
  static constexpr auto compute_type = CUTENSOR_COMPUTE_32F;
};
template <>
struct CUDATypeT<El::Complex<double>>
{
  typedef El::Complex<double> scalar_type;
  static constexpr auto value = CUDA_C_64F;
  static constexpr auto compute_type = CUTENSOR_COMPUTE_64F;
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

static inline cutensor::ModesType make_modes(size_t const ndims)
{
  std::vector<int32_t> modes(ndims + 1); // Add the sample dim.
  std::iota(begin(modes), end(modes), static_cast<int>('a'));
  return modes;
}

template <typename DataT>
static std::string get_desc_key(El::Matrix<DataT, El::Device::GPU> const& mat,
                                cutensor::DimsType const& dims_in)
{
  auto const& dims = dims_in.get();
  std::ostringstream oss;
  oss << mat.Height() << "," << mat.Width() << "," << mat.LDim() << ";";
  if (dims.size() == 0) {
    oss << "scalar";
  }
  else {
    oss << dims.front();
    for (size_t ii = 1; ii < dims.size(); ++ii)
      oss << "," << dims[ii];
  }
  oss << ";" << lbann::TypeName<DataT>();
  return oss.str();
}

template <typename DataT>
static cutensorTensorDescriptor_t
get_descriptor(El::Matrix<DataT, El::Device::GPU> const& mat,
               cutensor::DimsType const& dims)
{
  /** @brief Keep track of descriptors so we don't have to repeatedly
   *         rebuild them.
   */
  static std::unordered_map<std::string, cutensorTensorDescriptor_t> s_desc_map;

  auto key = get_desc_key(mat, dims); // captures Width to account for
                                      // minibatch size and LDim to
                                      // account for stride.
  auto iter = s_desc_map.find(key);
  if (iter == end(s_desc_map)) {
    std::vector<int64_t> extents = dims.get();
    extents.push_back(mat.Width()); // Don't forget MB size

    auto strides = get_strides(dims);
    strides.get().push_back(mat.LDim()); // Don't forget sample stride.

    cutensorTensorDescriptor_t desc;
    CHECK_CUTENSOR(cutensorInitTensorDescriptor(get_handle_ptr(),
                                                &desc,
                                                extents.size(),
                                                extents.data(),
                                                strides.get().data(),
                                                CUDAType<DataT>,
                                                CUTENSOR_OP_IDENTITY));
    s_desc_map.emplace(std::move(key), desc);
    return desc;
  }
  return iter->second;
}

} // namespace lbann

#endif // LBANN_UTILS_CUTENSOR_SUPPORT_HPP_INCLUDED
