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

#ifndef LBANN_LAYER_WEIGHTS_HPP_INCLUDED
#define LBANN_LAYER_WEIGHTS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Output a weights tensor
 *
 *  Interfaces with a @c weights object and outputs its tensor.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class weights_layer : public data_type_layer<TensorDataType>
{

public:
  /** @name Public Types */
  ///@{

  /** @brief The concrete local matrix type used by this object. */
  using MatType = El::Matrix<TensorDataType, Device>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  weights_layer(std::vector<El::Int> dims = {});
  weights_layer(const weights_layer& other) = default;
  weights_layer& operator=(const weights_layer& other) = default;
  weights_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS | WEIGHTS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;
};

// Builder function

// Explicit template instantiation
#ifndef LBANN_WEIGHTS_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class weights_layer<T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class weights_layer<T, data_layout::MODEL_PARALLEL, Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_WEIGHTS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_WEIGHTS_HPP_INCLUDED
