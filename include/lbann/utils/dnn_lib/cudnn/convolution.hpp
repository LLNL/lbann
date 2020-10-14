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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_CONVOLUTION_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_CONVOLUTION_HPP_

#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

/** @brief Which forward convolution algorithm to use. */
enum class fwd_conv_alg
{
  IMPLICIT_GEMM,
  IMPLICIT_PRECOMP_GEMM,
  GEMM,
  DIRECT,
  FFT,
  FFT_TILING,
  WINOGRAD,
  WINOGRAD_NONFUSED,
};// enum class fwd_conv_alg

/** @brief Which backward convolution algorithm to use. */
enum class bwd_conv_alg
{
  GEMM, // "ALGO_0" in cuDNN (are the renamed correct?)
  DIRECT, // "ALGO_1" in cuDNN
  FFT,
  FFT_TILING,
  WINOGRAD,
  WINOGRAD_NONFUSED,
};// enum class bwd_conv_alg

/** @brief Which backward convolution filter algorithm to use. */
enum class bwd_conv_filter
{
  ALGO_0, // need a better name
  ALGO_1, // need a better name
  FFT,
  ALGO_3, // need a better name
  WINOGRAD_NONFUSED,
  FFT_TILING,
};// enum class bwd_conv_filter

#ifdef LBANN_HAS_CUDNN  
namespace cudnn
{

/** @brief Convert a LBANN forward convolution algorithm to the cuDNN
 * equivalent value. */
inline cudnnConvolutionFwdAlgo_t to_cudnn(fwd_conv_alg a)
{
  switch (a)
  {
  case fwd_conv_alg::IMPLICIT_GEMM: return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  case fwd_conv_alg::IMPLICIT_PRECOMP_GEMM: return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  case fwd_conv_alg::GEMM: return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  case fwd_conv_alg::DIRECT: return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
  case fwd_conv_alg::FFT: return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  case fwd_conv_alg::FFT_TILING: return CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
  case fwd_conv_alg::WINOGRAD: return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  case fwd_conv_alg::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a cuDNN forward convolution algorithm to the LBANN
 * equivalent value. */
inline fwd_conv_alg from_cudnn(cudnnConvolutionFwdAlgo_t a)
{
  switch (a)
  {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return fwd_conv_alg::IMPLICIT_GEMM;
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return fwd_conv_alg::IMPLICIT_PRECOMP_GEMM;
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return fwd_conv_alg::GEMM;
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return fwd_conv_alg::DIRECT;
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return fwd_conv_alg::FFT;
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return fwd_conv_alg::FFT_TILING;
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return fwd_conv_alg::WINOGRAD;
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return fwd_conv_alg::WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution algorithm to the cuDNN
 * equivalent value. */
inline cudnnConvolutionBwdDataAlgo_t to_cudnn(bwd_conv_alg a)
{
  switch (a)
  {
  case bwd_conv_alg::GEMM: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  case bwd_conv_alg::DIRECT: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  case bwd_conv_alg::FFT: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
  case bwd_conv_alg::FFT_TILING: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
  case bwd_conv_alg::WINOGRAD: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
  case bwd_conv_alg::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a cuDNN backward convolution algorithm to the LBANN
 * equivalent value. */
inline bwd_conv_alg from_cudnn(cudnnConvolutionBwdDataAlgo_t a)
{
  switch (a)
  {
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: return bwd_conv_alg::GEMM;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: return bwd_conv_alg::DIRECT;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: return bwd_conv_alg::FFT;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: return bwd_conv_alg::FFT_TILING;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: return bwd_conv_alg::WINOGRAD;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return bwd_conv_alg::WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution filter algorithm to the cuDNN
 * equivalent value. */
inline cudnnConvolutionBwdFilterAlgo_t to_cudnn(bwd_conv_filter a)
{
  switch (a)
  {
  case bwd_conv_filter::ALGO_0: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  case bwd_conv_filter::ALGO_1: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  case bwd_conv_filter::FFT: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
  case bwd_conv_filter::ALGO_3: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
  case bwd_conv_filter::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
  case bwd_conv_filter::FFT_TILING: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

/** @brief Convert a cuDNN backward convolution filter algorithm to the LBANN
 * equivalent value. */
inline bwd_conv_filter from_cudnn(cudnnConvolutionBwdFilterAlgo_t a)
{
  switch (a)
  {
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0: return bwd_conv_filter::ALGO_0;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1: return bwd_conv_filter::ALGO_1;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: return bwd_conv_filter::FFT;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3: return bwd_conv_filter::ALGO_3;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: return bwd_conv_filter::WINOGRAD_NONFUSED;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING: return bwd_conv_filter::FFT_TILING;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_forward(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  ConvolutionDescriptor const& convDesc,
  fwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& yDesc,
  El::AbstractMatrix<TensorDataType>& y,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionForward(handle_manager.get(),
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      wDesc,
                                      w.LockedBuffer(),
                                      convDesc,
                                      to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      yDesc,
                                      y.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_forward(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  ConvolutionDescriptor const& convDesc,
  fwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& yDesc,
  El::AbstractMatrix<TensorDataType>& y)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(y),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(w),
                                     gpu::get_sync_info(x));
  convolution_forward(alpha_in, xDesc, x, wDesc, w, convDesc, alg,
                      workSpace, beta_in, yDesc, y, multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward(
  ScalarParameterType const& alpha_in,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dxDesc,
  El::AbstractMatrix<TensorDataType>& dx,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionBackwardData(handle_manager.get(),
                                      &alpha,
                                      wDesc,
                                      w.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      dxDesc,
                                      dx.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward(
  ScalarParameterType const& alpha_in,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dxDesc,
  El::AbstractMatrix<TensorDataType>& dx)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(w));
  convolution_backward(alpha_in, wDesc, w, dyDesc, dy, convDesc, alg,
                       workSpace, beta_in, dxDesc, dx, multisync);

}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_bias(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dbDesc,
  El::AbstractDistMatrix<TensorDataType>& db,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionBackwardBias(handle_manager.get(),
                                           &alpha,
                                           dyDesc,
                                           dy.LockedBuffer(),
                                           &beta,
                                           dbDesc,
                                           db.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_bias(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dbDesc,
  El::AbstractDistMatrix<TensorDataType>& db)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(db),
                                     gpu::get_sync_info(dy));
  convolution_backward_bias(alpha_in, dyDesc, dy,
                            beta_in, dbDesc, db,
                            multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_filter(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_conv_filter alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  FilterDescriptor const& dwDesc,
  El::AbstractDistMatrix<TensorDataType>& dw,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionBackwardFilter(handle_manager.get(),
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      dwDesc,
                                      dw.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_filter(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_conv_filter alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  FilterDescriptor const& dwDesc,
  El::AbstractDistMatrix<TensorDataType>& dw)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dw),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(x));
  convolution_backward_filter(alpha_in, xDesc, x, dyDesc, dy, convDesc, alg,
                              workSpace, beta_in, dwDesc, dw, multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void add_tensor(ScalarParameterType const& alpha_in,
                TensorDescriptor const& aDesc,
                El::AbstractDistMatrix<TensorDataType> const& A,
                ScalarParameterType const& beta_in,
                TensorDescriptor const& cDesc,
                El::AbstractMatrix<TensorDataType>& C,
                El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnAddTensor(handle_manager.get(),
                             &alpha,
                             aDesc,
                             A.LockedBuffer(),
                             &beta,
                             cDesc,
                             C.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void add_tensor(ScalarParameterType const& alpha_in,
                TensorDescriptor const& aDesc,
                El::AbstractDistMatrix<TensorDataType> const& A,
                ScalarParameterType const& beta_in,
                TensorDescriptor const& cDesc,
                El::AbstractMatrix<TensorDataType>& C)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(C),
                                     gpu::get_sync_info(A));
  add_tensor(alpha_in, aDesc, A, beta_in, cDesc, C, multisync);
}

}// namespace cudnn
#endif // LBANN_HAS_CUDNN
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_CONVOLUTION_HPP_
