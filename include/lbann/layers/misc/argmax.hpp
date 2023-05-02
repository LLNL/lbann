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

#ifndef LBANN_LAYERS_MISC_ARGMAX_HPP_INCLUDED
#define LBANN_LAYERS_MISC_ARGMAX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Get index of maximum-value tensor entry
 *
 *  Expects a 1D input tensor. If multiple entries have the same
 *  maximum value, outputs the index of the first one.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class argmax_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "argmax layer only supports data parallel layout");
  static_assert(Device == El::Device::CPU, "argmax layer only supports CPU");

public:
  argmax_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm) {}
  argmax_layer* copy() const override { return new argmax_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "argmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  argmax_layer() : argmax_layer(nullptr) {}

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;
};

#ifndef LBANN_ARGMAX_LAYER_INSTANTIATE
#define PROTO(T)                                                               \
  extern template class argmax_layer<T,                                        \
                                     data_layout::DATA_PARALLEL,               \
                                     El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#endif // LBANN_ARGMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_ARGMAX_HPP_INCLUDED
