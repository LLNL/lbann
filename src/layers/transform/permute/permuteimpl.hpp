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
#ifndef LBANN_SRC_LAYERS_TRANSFORM_PERMUTE_PERMUTEIMPL_HPP_INCLUDED
#define LBANN_SRC_LAYERS_TRANSFORM_PERMUTE_PERMUTEIMPL_HPP_INCLUDED

#include "lbann/layers/transform/permute.hpp"

#ifdef LBANN_HAS_CUTENSOR
#include "cutensor_permuteimpl.hpp"
#endif

#if defined(LBANN_HAS_CUTT) || defined(LBANN_HAS_HIPTT)
#include "cutt_permuteimpl.hpp"
#endif

#include "lbann/utils/tensor_dims_utils.hpp"

#include <cereal/cereal.hpp>

namespace lbann {

template <typename T>
class PermuteLayer<T>::PermuteImpl
{
public:
#ifdef LBANN_HAS_CUTENSOR
  using DeviceImplType = cuTENSOR_PermuteImpl;
#elif defined(LBANN_HAS_CUTT) || defined(LBANN_HAS_HIPTT)
  using DeviceImplType = cuTT_PermuteImpl;
#endif // LBANN_HAS_CU{TT,TENSOR}
  using MatType = El::Matrix<T, El::Device::GPU>;
  using DimsType = typename DeviceImplType::DimsType;

public:
  // LBANN uses row-major tensor ordering.
  PermuteImpl(std::vector<int> const& perm_row_major);
  PermuteImpl(PermuteImpl const& other) = default;
  PermuteImpl(PermuteImpl&& other) = default;

  // Returns the row-major output dims.
  std::vector<int> setup_dims(std::vector<int> const& input_dims);

  void forward_prop(MatType const& prev_acts, MatType& acts) const;

  // Activations don't actually matter here...
  void backward_prop(MatType const& grad_wrt_out, MatType& grad_wrt_in);

  std::vector<int> get_perm() const;
  std::string describe_perm() const;

  void swap(PermuteImpl& other);

  // Serialization

  template <typename ArchiveT>
  void save(ArchiveT& ar) const;

  template <typename ArchiveT>
  void load(ArchiveT& ar);

  template <typename ArchiveT>
  static void load_and_construct(
    ArchiveT& ar,
    cereal::construct<PermuteLayer<T>::PermuteImpl>& construct);

private:
  DeviceImplType m_device_impl;

}; // class PermuteImpl

} // namespace lbann
#endif // LBANN_SRC_LAYERS_TRANSFORM_PERMUTE_PERMUTEIMPL_HPP_INCLUDED
