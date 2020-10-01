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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_LRN_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_LRN_HPP_

#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

/** @brief Internal LBANN names for supported LRN layer modes.  */
// Only one implemented in cudnn currently:
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnLRNCrossChannelForward
enum class lrn_mode
{
  CROSS_CHANNEL_DIM1,
};// enum class lrn_mode

namespace cudnn
{

/** @brief Convert an LBANN lrn_mode to the cuDNN equivalent value. */
inline cudnnLRNMode_t to_cudnn(lrn_mode mode)
{
  switch (mode)
  {
  case lrn_mode::CROSS_CHANNEL_DIM1: return CUDNN_LRN_CROSS_CHANNEL_DIM1;
  default:
    LBANN_ERROR("Invalid LRN layer mode requested.");
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_forward(LRNDescriptor normDesc,
                               ScalarParameterType const& alpha_in,
                               TensorDescriptor const& xDesc,
                               El::AbstractMatrix<TensorDataType> const& x,
                               ScalarParameterType const& beta_in,
                               TensorDescriptor const& yDesc,
                               El::AbstractMatrix<TensorDataType>& y,
                               El::SyncInfo<El::Device::GPU> const& si,
                               lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnLRNCrossChannelForward(handle_manager.get(),
                                          normDesc,
                                          to_cudnn(mode),
                                          &alpha,
                                          xDesc,
                                          x.LockedBuffer(),
                                          &beta,
                                          yDesc,
                                          y.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_forward(LRNDescriptor normDesc,
                               ScalarParameterType const& alpha_in,
                               TensorDescriptor const& xDesc,
                               El::AbstractMatrix<TensorDataType> const& x,
                               ScalarParameterType const& beta_in,
                               TensorDescriptor const& yDesc,
                               El::AbstractMatrix<TensorDataType>& y,
                               lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(x),
                                     gpu::get_sync_info(y));
  lrn_cross_channel_forward(normDesc,
                            alpha_in, xDesc, x,
                            beta_in, yDesc, y,
                            multisync, mode);
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_backward(LRNDescriptor normDesc,
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
                                El::SyncInfo<El::Device::GPU> const& si,
                                lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnLRNCrossChannelBackward(handle_manager.get(),
                                           normDesc,
                                           to_cudnn(mode),
                                           &alpha,
                                           yDesc,
                                           y.LockedBuffer(),
                                           dyDesc,
                                           dy.LockedBuffer(),
                                           xDesc,
                                           x.LockedBuffer(),
                                           &beta,
                                           dxDesc,
                                           dx.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_backward(LRNDescriptor normDesc,
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
                                lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(y),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(x),
                                     gpu::get_sync_info(dx));
  lrn_cross_channel_backward(normDesc,
                             alpha_in, yDesc, y, dyDesc, dy,
                             xDesc, x, beta_in, dxDesc, dx,
                             multisync, mode);
}

}// namespace cudnn
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_LRN_HPP_
