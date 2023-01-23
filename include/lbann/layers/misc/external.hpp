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

#ifndef LBANN_LAYERS_MISC_EXTERNAL_HPP_INCLUDED
#define LBANN_LAYERS_MISC_EXTERNAL_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Call external function
 *
 *  Expects any number of input tensors. Invokes a shared object (e.g., .so file)
 *  to call the layer.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class external_layer : public data_type_layer<TensorDataType> {
public:

  external_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm) { 
    // TODO: dlopen
  }
  virtual ~external_layer();
  external_layer* copy() const override { return new external_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "external"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

protected:

  friend class cereal::access;
  external_layer()
    : external_layer(nullptr)
  {}

  external_layer(external_layer const& other) {
    this->comm = other.comm;
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    // TODO
    //this->set_output_dims({1});
  }

  void fp_compute() override;
  void bp_compute() override;
};


#ifndef LBANN_EXTERNAL_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class external_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class external_layer<T, data_layout::MODEL_PARALLEL, Device>


#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_EXTERNAL_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_EXTERNAL_HPP_INCLUDED
