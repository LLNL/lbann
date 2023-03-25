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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_SOFTMAX_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_SOFTMAX_HPP_

#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "utils.hpp"

namespace lbann {

#if defined LBANN_HAS_CUDNN
namespace dnn_lib {

using namespace cudnn;

template <typename TensorDataType, typename ScalarParameterType>
void softmax_forward(ScalarParameterType const& alpha_in,
                     TensorDescriptor const& xDesc,
                     El::Matrix<TensorDataType, El::Device::GPU> const& x,
                     ScalarParameterType const& beta_in,
                     TensorDescriptor const& yDesc,
                     El::Matrix<TensorDataType, El::Device::GPU>& y,
                     softmax_mode mode,
                     softmax_alg alg = softmax_alg::ACCURATE)
{
  // Short-circuit if we can
  if (x.IsEmpty())
    return;

  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto multisync =
    El::MakeMultiSync(gpu::get_sync_info(y), gpu::get_sync_info(x));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnSoftmaxForward(handle_manager.get(),
                                  cudnn::to_cudnn(alg),
                                  cudnn::to_cudnn(mode),
                                  &alpha,
                                  xDesc,
                                  x.LockedBuffer(),
                                  &beta,
                                  yDesc,
                                  y.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void softmax_backward(ScalarParameterType const& alpha_in,
                      TensorDescriptor const& yDesc,
                      El::Matrix<TensorDataType, El::Device::GPU> const& y,
                      TensorDescriptor const& dyDesc,
                      El::Matrix<TensorDataType, El::Device::GPU> const& dy,
                      ScalarParameterType const& beta_in,
                      TensorDescriptor const& dxDesc,
                      El::Matrix<TensorDataType, El::Device::GPU>& dx,
                      softmax_mode mode,
                      softmax_alg alg = softmax_alg::ACCURATE)
{
  // Short-circuit if we can
  if (y.IsEmpty())
    return;

  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx),
                                     gpu::get_sync_info(y),
                                     gpu::get_sync_info(dy));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnSoftmaxBackward(handle_manager.get(),
                                   cudnn::to_cudnn(alg),
                                   cudnn::to_cudnn(mode),
                                   &alpha,
                                   yDesc,
                                   y.LockedBuffer(),
                                   dyDesc,
                                   dy.LockedBuffer(),
                                   &beta,
                                   dxDesc,
                                   dx.Buffer()));
}

} // namespace dnn_lib
#endif // LBANN_HAS_CUDNN
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_SOFTMAX_HPP_
