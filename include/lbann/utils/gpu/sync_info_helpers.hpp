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

#ifndef LBANN_UTILS_GPU_SYNC_INFO_HELPERS_HPP_INCLUDED
#define LBANN_UTILS_GPU_SYNC_INFO_HELPERS_HPP_INCLUDED

#include <El.hpp>

namespace lbann {
namespace gpu {

/** @name SyncInfo extractors. */
///@{
/** @brief Get a SyncInfo from an AbstractMatrix.
 *
 *  @throws std::bad_cast If @c m is not a GPU matrix.
 */
template <typename TensorDataType>
El::SyncInfo<El::Device::GPU> get_sync_info(
  El::AbstractMatrix<TensorDataType> const& m) {
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;
  return El::SyncInfoFromMatrix(dynamic_cast<GPUMatType const&>(m));
}

/** @brief Get a SyncInfo from an Matrix. */
template <typename TensorDataType>
El::SyncInfo<El::Device::GPU> get_sync_info(
  El::Matrix<TensorDataType, El::Device::GPU> const& m) noexcept {
  return El::SyncInfoFromMatrix(m);
}

/** @brief Get a SyncInfo from an AbstractDistMatrix.
 *
 *  @throws std::bad_cast If @c m is not a GPU matrix.
 */
template <typename TensorDataType>
El::SyncInfo<El::Device::GPU> get_sync_info(
  El::AbstractDistMatrix<TensorDataType> const& m) {
  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;
  return El::SyncInfoFromMatrix(
    dynamic_cast<GPUMatType const&>(m.LockedMatrix()));
}

/** @brief Get a SyncInfo from a DistMatrix.
 *
 *  This saves a dynamic_cast over the AbstractDistMatrix version.
 */
template <typename TensorDataType, El::Dist RowDist, El::Dist ColDist>
El::SyncInfo<El::Device::GPU> get_sync_info(
  El::DistMatrix<TensorDataType,
                 RowDist, ColDist,
                 El::ELEMENT,
                 El::Device::GPU> const& m) noexcept {
  return El::SyncInfoFromMatrix(m.LockedMatrix());
}

} // namespace gpu
} // namespace lbann
#endif // LBANN_UTILS_GPU_SYNC_INFO_HELPERS_HPP_INCLUDED
