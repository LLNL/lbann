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

#ifndef LBANN_UTILS_SYNC_INFO_HELPERS_HPP_INCLUDED
#define LBANN_UTILS_SYNC_INFO_HELPERS_HPP_INCLUDED

#include <El.hpp>

namespace lbann {

/** @name SyncInfo extractors. */
///@{

/** @brief Get a SyncInfo from an Matrix. */
template <typename TensorDataType, El::Device D>
El::SyncInfo<D> get_sync_info(
  El::Matrix<TensorDataType, D> const& m) noexcept
{
  return El::SyncInfoFromMatrix(m);
}

/** @brief Get a SyncInfo from a DistMatrix.
 *
 *  This saves a dynamic_cast over the AbstractDistMatrix version.
 */
template <typename TensorDataType,
          El::Dist RowDist,
          El::Dist ColDist,
          El::Device D>
El::SyncInfo<D> get_sync_info(
  El::DistMatrix<TensorDataType,
                 RowDist, ColDist,
                 El::ELEMENT,
                 D> const& m) noexcept
{
  return El::SyncInfoFromMatrix(m.LockedMatrix());
}

///@}

/** @brief Force the MultiSync to the master SyncInfo
 *  @details This is a short-hand for static_casting for cases in
 *           which implicit conversion clashes with template
 *           deduction, for example.
 */
template <El::Device D, El::Device... Ds>
inline auto force(El::MultiSync<D, Ds...> const& x)
  -> El::SyncInfo<D> const&
{
  return x;
}

} // namespace lbann
#endif // LBANN_UTILS_SYNC_INFO_HELPERS_HPP_INCLUDED
