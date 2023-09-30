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

#ifndef LBANN_LAYERS_MISC_UNIFORM_HASH_HPP_INCLUDED
#define LBANN_LAYERS_MISC_UNIFORM_HASH_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Apply a hash function to get uniformly distributed values
 *
 *  Each input entry is hashed with MD5 and scaled to [0,1).
 *
 *  @warning Currently only supported on GPU.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class uniform_hash_layer : public data_type_layer<TensorDataType>
{
#ifdef LBANN_HAS_GPU
  static_assert(Device == El::Device::GPU,
                "uniform_hash_layer only supports GPU");
#else
  static_assert(Device != El::Device::CPU, /** @todo Make nicer */
                "uniform_hash_layer only supports GPU");
#endif // LBANN_HAS_GPU

public:
  uniform_hash_layer(lbann_comm* comm);

  uniform_hash_layer(const uniform_hash_layer& other) = default;
  uniform_hash_layer& operator=(const uniform_hash_layer& other) = default;
  uniform_hash_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  uniform_hash_layer() : uniform_hash_layer(nullptr) {}

  void setup_dims() override;

  void fp_compute() override;
};

template <typename T, data_layout L, El::Device D>
void uniform_hash_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_uniform_hash();
}

#ifdef LBANN_HAS_GPU
#ifndef LBANN_UNIFORM_HASH_LAYER_INSTANTIATE
#define PROTO(T)                                                               \
  extern template class uniform_hash_layer<T,                                  \
                                           data_layout::DATA_PARALLEL,         \
                                           El::Device::GPU>;                   \
  extern template class uniform_hash_layer<T,                                  \
                                           data_layout::MODEL_PARALLEL,        \
                                           El::Device::GPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_UNIFORM_HASH_LAYER_INSTANTIATE
#endif // LBANN_HAS_GPU

} // namespace lbann

#endif // LBANN_LAYERS_MISC_UNIFORM_HASH_HPP_INCLUDED
