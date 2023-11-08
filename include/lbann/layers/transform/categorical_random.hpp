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

#ifndef LBANN_LAYER_CATEGORICAL_RANDOM_HPP_INCLUDED
#define LBANN_LAYER_CATEGORICAL_RANDOM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random categorical outputs.
 *
 *  Inputs are probability distributions and outputs are one-hot
 *  vectors. An input entry is the probability that the corresponding
 *  output entry is one.
 *
 *  @todo Remove.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class categorical_random_layer : public data_type_layer<TensorDataType>
{
  static_assert(Dev == El::Device::CPU,
                "categorical random layer currently only supports CPU");
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "categorical random layer currently only "
                "supports DATA_PARALLEL");

public:
  categorical_random_layer(lbann_comm* comm)
    : data_type_layer<TensorDataType>(comm)
  {}
  categorical_random_layer* copy() const override
  {
    return new categorical_random_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "categorical random"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  categorical_random_layer() : categorical_random_layer(nullptr) {}

  void fp_compute() override;
};

#ifndef LBANN_CATEGORICAL_RANDOM_LAYER_INSTANTIATE

#define PROTO(T)                                                               \
  extern template class categorical_random_layer<T,                            \
                                                 data_layout::DATA_PARALLEL,   \
                                                 El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF

#endif // LBANN_CATEGORICAL_RANDOM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_CATEGORICAL_RANDOM_HPP_INCLUDED
