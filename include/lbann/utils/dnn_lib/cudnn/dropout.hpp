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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_DROPOUT_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_DROPOUT_HPP_

#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

#if defined LBANN_HAS_CUDNN
namespace cudnn
{

inline size_t get_dropout_states_size()
{
  size_t size;
  CHECK_CUDNN(cudnnDropoutGetStatesSize(get_handle(), &size));
  return size;
}

inline size_t get_dropout_reserve_space_size(TensorDescriptor xDesc)
{
  size_t size;
  CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(xDesc, &size));
  return size;
}

template <typename TensorDataType>
void dropout_forward(DropoutDescriptor dropoutDesc,
                     TensorDescriptor xDesc,
                     El::AbstractMatrix<TensorDataType> const& x,
                     TensorDescriptor yDesc,
                     El::AbstractMatrix<TensorDataType>& y,
                     El::AbstractMatrix<TensorDataType>& workSpace,
                     El::SyncInfo<El::Device::GPU> const& si)
{
  auto handle_manager = internal::make_default_handle_manager(si);
  CHECK_CUDNN(
    cudnnDropoutForward(handle_manager.get(),
                        dropoutDesc,
                        xDesc,
                        x.LockedBuffer(),
                        yDesc,
                        y.Buffer(),
                        workSpace.Buffer(),
                        workSpace.Height() * sizeof(TensorDataType)));
}

template <typename TensorDataType>
void dropout_forward(DropoutDescriptor dropoutDesc,
                     TensorDescriptor xDesc,
                     El::AbstractMatrix<TensorDataType> const& x,
                     TensorDescriptor yDesc,
                     El::AbstractMatrix<TensorDataType>& y,
                     El::AbstractMatrix<TensorDataType>& workSpace)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(y),
                                     gpu::get_sync_info(x));
  dropout_forward(dropoutDesc, xDesc, x, yDesc, y, workSpace, multisync);
}

template <typename TensorDataType>
void dropout_backward(DropoutDescriptor dropoutDesc,
                      TensorDescriptor yDesc,
                      El::AbstractMatrix<TensorDataType> const& y,
                      TensorDescriptor xDesc,
                      El::AbstractMatrix<TensorDataType>& x,
                      El::AbstractMatrix<TensorDataType>& workSpace,
                      El::SyncInfo<El::Device::GPU> const& si)
{
  auto handle_manager = internal::make_default_handle_manager(si);
  CHECK_CUDNN(
    cudnnDropoutBackward(handle_manager.get(),
                         dropoutDesc,
                         yDesc,
                         y.LockedBuffer(),
                         xDesc,
                         x.Buffer(),
                         workSpace.Buffer(),
                         workSpace.Height() * sizeof(TensorDataType)));
}

template <typename TensorDataType>
void dropout_backward(DropoutDescriptor dropoutDesc,
                      TensorDescriptor yDesc,
                      El::AbstractMatrix<TensorDataType> const& y,
                      TensorDescriptor xDesc,
                      El::AbstractMatrix<TensorDataType>& x,
                      El::AbstractMatrix<TensorDataType>& workSpace)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(x),
                                     gpu::get_sync_info(y));
  dropout_backward(dropoutDesc, yDesc, y, xDesc, x, workSpace, multisync);
}

}// namespace cudnn
#endif // LBANN_HAS_CUDNN
  
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_DROPOUT_HPP_
