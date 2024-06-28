////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UTILS_DNN_LIB_MIOPEN_TRANSFORM_TENSOR_HPP_
#define LBANN_UTILS_DNN_LIB_MIOPEN_TRANSFORM_TENSOR_HPP_

#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann {

#ifdef LBANN_HAS_MIOPEN
namespace dnn_lib {

using namespace miopen;

template <typename TensorDataType, typename ScalarParameterType>
void transform_tensor(ScalarParameterType const& alpha_in,
                      TensorDescriptor const& xDesc,
                      const TensorDataType* x,
                      ScalarParameterType const& beta_in,
                      TensorDescriptor const& yDesc,
                      TensorDataType* y,
                      dnnHandle_t handle)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto alpha = static_cast<LibScalingParamT>(alpha_in);
  auto beta = static_cast<LibScalingParamT>(beta_in);
  CHECK_MIOPEN(
    miopenTransformTensor(handle, &alpha, xDesc, x, &beta, yDesc, y));
}

} // namespace dnn_lib
#endif // LBANN_HAS_MIOPEN
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_MIOPEN_TRANSFORM_TENSOR_HPP_
