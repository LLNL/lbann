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

#ifndef LBANN_LAYERS_TRANSFORM_PERMUTE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_PERMUTE_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Permute the indices of a tensor.
 *
 *  Expects one input tensor of order N, and a length N array of
 *  permuted indices [0..N-1], with respect to the input tensor
 *  dimensions. (Therefore, passing axes=[0,1,2] for a rank-3 tensor
 *  is just a (deep) copy.)
 *
 *  At this time, only simple "tensor transpose" is supported. Each
 *  index must be accounted for in the permuted array.
 *
 *  The current implementation of this layer is written for
 *  CUDA. Other implementations will be added as needed.
 */
template <typename T>
class PermuteLayer final : public data_type_layer<T>
{
public:
  /** @name Lifetime management */
  ///@{

  PermuteLayer(std::vector<int> const& axes);
  PermuteLayer(PermuteLayer const& other);
  PermuteLayer& operator=(PermuteLayer const& other);
  PermuteLayer(PermuteLayer&& other) = default;
  PermuteLayer& operator=(PermuteLayer&& other) = default;
  ~PermuteLayer();

  PermuteLayer* copy() const final;
  void swap(PermuteLayer& other);

  ///@}
  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const final;
  data_layout get_data_layout() const final;
  El::Device get_device_allocation() const final;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }
  description get_description() const final;

protected:
  friend class cereal::access;
  PermuteLayer();

  void setup_dims() final;
  void fp_compute() final;
  void bp_compute() final;

  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  class PermuteImpl;
  std::unique_ptr<PermuteImpl> m_impl;
};

#if defined(LBANN_HAS_TENSOR_PERMUTE)
// No member of this class will actually be instantiated unless LBANN
// is built with support for a tensor permute backend.

#ifndef LBANN_PERMUTE_LAYER_INSTANTIATE
#define PROTO(T) extern template class PermuteLayer<T>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_PERMUTE_LAYER_INSTANTIATE

#endif // LBANN_HAS_TENSOR_PERMUTE

} // namespace lbann
#endif // LBANN_LAYERS_TRANSFORM_PERMUTE_HPP_INCLUDED
