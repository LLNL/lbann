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

#ifndef LBANN_LAYERS_MISC_MINI_BATCH_SIZE_HPP_INCLUDED
#define LBANN_LAYERS_MISC_MINI_BATCH_SIZE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Mini-batch size.
 *
 *  Output tensor is a 1D tensor with a single entry containing the
 *  model's current mini-batch size.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class mini_batch_size_layer : public data_type_layer<TensorDataType>
{
public:
  mini_batch_size_layer(lbann_comm* comm);
  mini_batch_size_layer* copy() const override;

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
  mini_batch_size_layer() : mini_batch_size_layer(nullptr) {}

  void setup_dims() override;
  void fp_setup_outputs(El::Int mini_batch_size) override;
  void fp_compute() override;

private:
  /** Mini-batch size. */
  El::Int m_mini_batch_size = 0;
};

#ifndef LBANN_MINI_BATCH_SIZE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class mini_batch_size_layer<T,                               \
                                              data_layout::DATA_PARALLEL,      \
                                              Device>;                         \
  extern template class mini_batch_size_layer<T,                               \
                                              data_layout::MODEL_PARALLEL,     \
                                              Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_MINI_BATCH_SIZE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_MINI_BATCH_SIZE_HPP_INCLUDED
