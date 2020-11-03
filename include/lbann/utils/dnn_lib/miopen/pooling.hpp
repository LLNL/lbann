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
#ifndef LBANN_UTILS_DNN_LIB_MIOPEN_POOLING_HPP_
#define LBANN_UTILS_DNN_LIB_MIOPEN_POOLING_HPP_

#include "lbann/utils/ml_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

#ifdef LBANN_HAS_MIOPEN
namespace dnn_lib
{

using namespace miopen;

template <typename TensorDataType, typename ScalarParameterType>
void pooling_forward(PoolingDescriptor const& poolingDesc,
                     ScalarParameterType const& alpha_in,
                     TensorDescriptor const& xDesc,
                     El::AbstractMatrix<TensorDataType> const& x,
                     ScalarParameterType const& beta_in,
                     TensorDescriptor const& yDesc,
                     El::AbstractMatrix<TensorDataType>& y,
                     El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
                     El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  if (workspace.Height() == 0 || workspace.Width() == 0) { // Training use-case
    CHECK_MIOPEN(miopenPoolingForward(handle_manager.get(),
                                      poolingDesc,
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      &beta,
                                      yDesc,
                                      y.Buffer(),
                                      true,
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType)));
  }
  else {                                                  // Inference use-case
    CHECK_MIOPEN(miopenPoolingForward(handle_manager.get(),
                                      poolingDesc,
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      &beta,
                                      yDesc,
                                      y.Buffer(),
                                      false,
                                      nullptr,
                                      0UL));
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void pooling_forward(PoolingDescriptor const& poolingDesc,
                     ScalarParameterType const& alpha_in,
                     TensorDescriptor const& xDesc,
                     El::AbstractMatrix<TensorDataType> const& x,
                     ScalarParameterType const& beta_in,
                     TensorDescriptor const& yDesc,
                     El::AbstractMatrix<TensorDataType>& y,
                     El::Matrix<TensorDataType, El::Device::GPU>& workSpace)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(y),
                                     gpu::get_sync_info(x));
  pooling_forward(poolingDesc,
                  alpha_in, xDesc, x,
                  beta_in, yDesc, y,
                  workspace,
                  multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void pooling_backward(PoolingDescriptor const& poolingDesc,
                      ScalarParameterType const& alpha_in,
                      TensorDescriptor const& yDesc,
                      El::AbstractMatrix<TensorDataType> const& y,
                      TensorDescriptor const& dyDesc,
                      El::AbstractMatrix<TensorDataType> const& dy,
                      TensorDescriptor const& xDesc,
                      El::AbstractMatrix<TensorDataType> const& x,
                      ScalarParameterType const& beta_in,
                      TensorDescriptor const& dxDesc,
                      El::AbstractMatrix<TensorDataType>& dx,
                      El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
                      El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  if (workspace.Height() == 0 || workspace.Width() == 0) { // Training use-case
    CHECK_MIOPEN(miopenPoolingBackward(handle_manager.get(),
                                       poolingDesc,
                                       &alpha,
                                       yDesc,
                                       y.LockedBuffer(),
                                       dyDesc,
                                       dy.LockedBuffer(),
                                       xDesc,
                                       x.LockedBuffer(),
                                       &beta,
                                       dxDesc,
                                       dx.Buffer(),
                                       true,
                                       workSpace.Buffer(),
                                       workSpace.Height()*sizeof(TensorDataType)));
  }
  else {                                                  // Inference use-case
    CHECK_MIOPEN(miopenPoolingBackward(handle_manager.get(),
                                       poolingDesc,
                                       &alpha,
                                       yDesc,
                                       y.LockedBuffer(),
                                       dyDesc,
                                       dy.LockedBuffer(),
                                       xDesc,
                                       x.LockedBuffer(),
                                       &beta,
                                       dxDesc,
                                       dx.Buffer(),
                                       false,
                                       nullptr,
                                       0UL));
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void pooling_backward(PoolingDescriptor const& poolingDesc,
                      ScalarParameterType const& alpha_in,
                      TensorDescriptor const& yDesc,
                      El::AbstractMatrix<TensorDataType> const& y,
                      TensorDescriptor const& dyDesc,
                      El::AbstractMatrix<TensorDataType> const& dy,
                      TensorDescriptor const& xDesc,
                      El::AbstractMatrix<TensorDataType> const& x,
                      ScalarParameterType const& beta_in,
                      TensorDescriptor const& dxDesc,
                      El::AbstractMatrix<TensorDataType>& dx,
                      El::Matrix<TensorDataType, El::Device::GPU>& workSpace)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dx),
                                     gpu::get_sync_info(x),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(y));
  pooling_backward(poolingDesc,
                   alpha_in, yDesc, y, dyDesc, dy,
                   xDesc, x, beta_in, dxDesc, dx,
                   workSpace,
                   multisync);
}

}// namespace dnn_lib
#endif // LBANN_HAS_MIOPEN
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_MIOPEN_POOLING_HPP_
