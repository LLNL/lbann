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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_UPSAMPLE_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_UPSAMPLE_HPP_

#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann {

#ifdef LBANN_HAS_CUDNN
namespace dnn_lib {

using namespace cudnn;

template <typename TensorDataType, typename ScalarParameterType>
void upsample_nearest_forward(PoolingDescriptor const& poolingDesc,
                              ScalarParameterType const& alpha_in,
                              TensorDescriptor const& xDesc,
                              TensorDataType const* x,
                              ScalarParameterType const& beta_in,
                              TensorDescriptor const& yDesc,
                              TensorDataType* y,
                              dnnHandle_t handle)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnPoolingBackward(handle,
                                   poolingDesc,
                                   &alpha,
                                   NULL,
                                   NULL,
                                   xDesc,
                                   x,
                                   NULL,
                                   NULL,
                                   &beta,
                                   yDesc,
                                   y));
}

template <typename TensorDataType, typename ScalarParameterType>
void upsample_nearest_forward(PoolingDescriptor const& poolingDesc,
                              ScalarParameterType const& alpha_in,
                              TensorDescriptor const& xDesc,
                              El::AbstractMatrix<TensorDataType> const& x,
                              ScalarParameterType const& beta_in,
                              TensorDescriptor const& yDesc,
                              El::AbstractMatrix<TensorDataType>& y)
{
  auto multisync =
    El::MakeMultiSync(gpu::get_sync_info(y), gpu::get_sync_info(x));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  upsample_nearest_forward(poolingDesc,
                           alpha_in,
                           xDesc,
                           x.LockedBuffer(),
                           beta_in,
                           yDesc,
                           y.Buffer(),
                           handle_manager.get());
}

template <typename TensorDataType, typename ScalarParameterType>
void upsample_nearest_backward(PoolingDescriptor const& poolingDesc,
                               ScalarParameterType const& alpha_in,
                               TensorDescriptor const& dyDesc,
                               TensorDataType const* dy,
                               ScalarParameterType const& beta_in,
                               TensorDescriptor const& dxDesc,
                               TensorDataType* dx,
                               dnnHandle_t handle)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnPoolingForward(handle,
                                  poolingDesc,
                                  &alpha,
                                  dyDesc,
                                  dy,
                                  &beta,
                                  dxDesc,
                                  dx));
}

template <typename TensorDataType, typename ScalarParameterType>
void upsample_nearest_backward(PoolingDescriptor const& poolingDesc,
                               ScalarParameterType const& alpha_in,
                               TensorDescriptor const& dyDesc,
                               El::AbstractMatrix<TensorDataType> const& dy,
                               ScalarParameterType const& beta_in,
                               TensorDescriptor const& dxDesc,
                               El::AbstractMatrix<TensorDataType>& dx)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx),
                                     gpu::get_sync_info(dy));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  upsample_nearest_backward(poolingDesc,
                            alpha_in,
                            dyDesc,
                            dy.LockedBuffer(),
                            beta_in,
                            dxDesc,
                            dx.Buffer(),
                            handle_manager.get());
}

} // namespace dnn_lib
#endif // LBANN_HAS_CUDNN
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_UPSAMPLE_HPP_
