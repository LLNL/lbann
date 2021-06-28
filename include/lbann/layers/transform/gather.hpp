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

#ifndef LBANN_LAYERS_TRANSFORM_GATHER_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_GATHER_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Gather values from specified tensor indices.
 *
 *  @f[
 *    y[i] = x[\text{ind}[i]]
 *  @f]
 *
 *  The first input tensor is the values and the second is the
 *  indices. The two input tensors must have the same number of
 *  dimensions, and the output tensor will have the same dimensions as
 *  the index tensor. If an index is out-of-range, the corresponding
 *  output is set to zero.
 *
 *  @todo Only flat tensors are currently supported. For higher-order
 *  tensors, PyTorch
 *  (https://pytorch.org/docs/stable/generated/torch.gather.html) and
 *  TensorFlow
 *  (https://www.tensorflow.org/api_docs/python/tf/gather_nd) will
 *  gather along a specified dimension.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class gather_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "gather layer only supports data parallel layout");
public:

  gather_layer();
  gather_layer(const gather_layer& other) = default;
  gather_layer& operator=(const gather_layer& other) = default;

  gather_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;
  void bp_compute() override;

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
gather_layer<TensorDataType,Layout,Device>::gather_layer()
  : data_type_layer<TensorDataType>(nullptr) {
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gather_layer<TensorDataType,Layout,Device>* gather_layer<TensorDataType,Layout,Device>::copy() const {
  return new gather_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string gather_layer<TensorDataType,Layout,Device>::get_type() const {
  return "gather";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout gather_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device gather_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType,Layout,Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Tensor dimensions
  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);
  // Only support 1D indices
  const auto is_indices_not_1D = input1_dims.size() != 1;
  
  // Only support 1D or 2D values
  const auto is_values_1D =  input0_dims.size() == 1;
  const auto is_values_2D = input0_dims.size() == 2; 
  
  if(is_values_1D){
    this->set_output_dims(input1_dims);
  }else{
    this->set_output_dims(std::vector<int>{input0_dims[0],input1_dims[0]});
  }
  
  auto dims_to_str = [] (const std::vector<int>& dims) -> std::string {
    std::ostringstream ss;
    for (size_t i=0; i<dims.size(); ++i) {
      ss << (i>0 ? "x" : "") << dims[i];
    }
    return ss.str();
  };

  // Make sure input tensors have supported numbers of dimensions




  if (is_indices_not_1D || !(is_values_1D || is_values_2D)) {
    const auto& parent0 = this->get_parent_layer(0);
    const auto& parent1 = this->get_parent_layer(1);
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "has input tensors with different numbers of dimensions ",
      "(",parent0.get_type()," layer \"",parent0.get_name(),"\" ",
      "outputs ",dims_to_str(input0_dims),", ",
      parent1.get_type()," layer \"",parent1.get_name(),"\" ",
      "outputs ",dims_to_str(input1_dims),")");
  }

  // Check that tensors are 1D
  /// @todo Support gathering from/into higher-order tensors
  if (!is_values_1D && !is_values_2D) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "attempted to gather from a ",input0_dims.size(),"-D tensor ",
      "(",dims_to_str(input0_dims),"), "
      "but the gather layer currently only supports ",
      "gathering from a 1-D oe 2-D tensor");
  }

}

LBANN_DEFINE_LAYER_BUILDER(gather);

#ifndef LBANN_GATHER_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                 \
  extern template class gather_layer<          \
    T, data_layout::DATA_PARALLEL, Device>;
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_GATHER_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_GATHER_HPP_INCLUDED
