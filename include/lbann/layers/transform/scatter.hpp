////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_TRANSFORM_SCATTER_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_SCATTER_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/proto/layers.pb.h"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#include "lbann/layers/transform/distconv/distconv_scatter.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
class scatter_distconv_adapter
  :  public data_type_distconv_adapter <TensorDataType>{
  public:
    using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

    scatter_distconv_adapter(Layer &layer) : data_type_distconv_adapter<TensorDataType>(layer){}
    virtual ~scatter_distconv_adapter() = default;

    void setup_distribution(tensor_overlap_constraints &constraints) override;
    void setup_layer(size_t workspace_capacity) override;
    void fp_compute();
    void bp_compute();

    std::unique_ptr<dc::Scatter<TensorDataType>> m_scatter_operator;
    size_t m_workspace_buffer_size{0};
  };
#endif // LBANN_HAS_DISTCONV

/** @brief Scatter values to specified tensor indices
 *
 *  Expects two input tensors: an @f$ N @f$-D data tensor and a 1D
 *  index vector. For 1D data:
 *  @f[
 *    y[\text{ind}[i]] = x[i]
 *  @f]
 *  Out-of-range indices are ignored.
 *
 *  For higher-dimensional data, the layer performs a scatter along
 *  one dimension. For example, with 2D data and axis=1,
 *  @f[
 *    y[i,\text{ind}[j]] = x[i,j]
 *  @f]
 *  Currently, only 1D and 2D data is supported.
 *
 *  The size of the index vector must match the size of the data
 *  tensor along the scatter dimension.
 *
 *  @todo Support higher-dimensional data
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class scatter_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "scatter layer only supports data parallel layout");
public:

  scatter_layer(const std::vector<int>& dims, const int axis);
  scatter_layer(const scatter_layer& other) = default;
  scatter_layer& operator=(const scatter_layer& other) = default;

  scatter_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

protected:

  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  scatter_layer()
    : scatter_layer({1},-1)
  {}
  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void fp_compute() override;
  void bp_compute() override;
#ifdef LBANN_HAS_DISTCONV
  friend class scatter_distconv_adapter<TensorDataType, Layout, Device>;
  void setup_distconv_adapter();
  bool is_distconv_supported() const override;
  scatter_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() override;
  const scatter_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
private:
  int m_scatter_axis;

};

// =========================================================
// Implementation
// =========================================================

template <typename T, data_layout L, El::Device D>
void scatter_layer<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_scatter();
  protobuf::assign_to_repeated(*msg->mutable_dims(), this->get_output_dims());
  msg->mutable_axis()->set_value(m_scatter_axis);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
scatter_layer<TensorDataType,Layout,Device>::scatter_layer(
  const std::vector<int>& dims, const int axis)
  : data_type_layer<TensorDataType>(nullptr),
    m_scatter_axis{axis} {
  this->m_expected_num_parent_layers = 2;
  this->set_output_dims(dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
scatter_layer<TensorDataType,Layout,Device>* scatter_layer<TensorDataType,Layout,Device>::copy() const {
  return new scatter_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string scatter_layer<TensorDataType,Layout,Device>::get_type() const {
  return "scatter";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout scatter_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device scatter_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void scatter_layer<TensorDataType,Layout,Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  

  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);
  const auto& output_dims = this->get_output_dims();

  auto dims_to_str = [] (const std::vector<int>& dims) -> std::string {
    std::ostringstream ss;
    for (size_t i=0; i<dims.size(); ++i) {
      ss << (i>0 ? "x" : "") << dims[i];
    }
    return ss.str();
  };

  // Tensor dimension requirements are different 
  // when using distconv 
  // Distconv requires 3D inputs for both values
  // and indices
  
  #ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()){
    const auto is_values_3D = input0_dims.size(); 
    const auto is_indices_3D = input1_dims.size();
    const auto is_output_3D = output_dims.size();
    
    // Matrices need to be 3D
    if (!is_values_3D || !is_indices_3D){
       
      LBANN_ERROR(this->get_type(), " Layer \"", this->get_name(),"\" ",
        "has values input (", dims_to_str(input0_dims),") ",
        "has indices input (", dims_to_str(input1_dims),"). ", 
        "Distconv requires both to be 3D. ");
    }
    // The 0-th dimension of the values and indices must match
    if (input0_dims[0] != input1_dims[0]){
      LBANN_ERROR(this->get_type(), " Layer \"", this->get_name(),"\" ",
        "has values input (", dims_to_str(input0_dims),") ",
        "has indices input (", dims_to_str(input1_dims),"). ", 
        "Distconv requires the 0-th dimension to match. ");
    }

    // The 1st and 2D dimension of the values and output must match
    const auto output_dim_fail = input0_dims[1] != output_dims[1] || 
        input0_dims[2] != output_dims[2];

    if (output_dim_fail){
      LBANN_ERROR(this->get_type(), " Layer \"", this->get_name(),"\" ",
        "has values input (", dims_to_str(input0_dims),") ",
        "has indices input (", dims_to_str(input1_dims),"). ", 
        "Distconv requires the 0-th dimension to match. ");
    }

    // Enable distconv only for scatter along the 0-th dimension
    if (this->m_scatter_axis != 0){
      LBANN_ERROR(this->get_type(), " Layer \"", this->get_name(),"\" ",
         "requires the scatter dimension to be 0 when using distconv");
    }

    return ;
  }
  #endif // LBANN_HAS_DISTCONV
  // Tensor dimensions
  // Check if value matrix is 1D or 2D

  const auto is_values_1D = input0_dims.size() == 1;
  const auto is_values_2D = input0_dims.size() == 2;

  // Check if output matrix is 1D or 2D

  const auto is_output_1D = output_dims.size() == 1;
  const auto is_output_2D = output_dims.size() == 2;

  if(is_values_2D){
    if(this->m_scatter_axis == -1){
      LBANN_ERROR(
        this->get_type(), " Layer \"", this->get_name(),"\" ",
        "has 2D input, but does not set a scatter axis.",
        " Axis must be either set to 0 or 1");
    }
  }
  // Make sure input tensors have same dimensions
  if (input0_dims != input1_dims) {

    // If input tensors are not same, make sure it's 2D and 1D
    const auto matching_dim = this->m_scatter_axis == 0? 0 : 1;
    if(input0_dims[matching_dim] != input1_dims[0]){
      const auto& parent0 = this->get_parent_layer(0);
      const auto& parent1 = this->get_parent_layer(1);
      LBANN_ERROR(
        this->get_type()," layer \"",this->get_name(),"\" ",
        "has input tensors with different outer dimensions ",
        "(",parent0.get_type()," layer \"",parent0.get_name(),"\" ",
        "outputs ",dims_to_str(input0_dims),", ",
        parent1.get_type()," layer \"",parent1.get_name(),"\" ",
        "outputs ",dims_to_str(input1_dims),")");
    }
  }

  // Check tensor dimensions
  if (input1_dims.size() != 1 ||
      !(is_values_1D || is_values_2D) ||
      input0_dims.size() != output_dims.size()) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "attempted to scatter from a ",input0_dims.size(),"-D tensor ",
      "(",dims_to_str(input0_dims),"), to a ", output_dims.size(),"-D tensor",
      "but the scatter layer currently only supports ",
      "scattering to and from a 1-D or 2-D tensor and the input and output tensors",
      "must have the same number of dimensions");
  }
  // Check if either output is 1D or the first dim matches for input and output
  if ( ! is_output_1D && (is_output_2D && output_dims[0] != input0_dims[0])) {
    const auto matching_dim = this->m_scatter_axis == 0? 1 : 0;
    if (output_dims[matching_dim] != input0_dims[matching_dim]){

      LBANN_ERROR(
        this->get_type()," layer \"",this->get_name(),"\" ",
        "attempted to scatter into a ",output_dims.size(),"-D tensor ",
        "(",dims_to_str(output_dims),"), "
        "but expected ", input0_dims[matching_dim], " on axis ",
        matching_dim);
      }
    }
}

#ifndef LBANN_SCATTER_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class scatter_layer<T, data_layout::DATA_PARALLEL, Device>;
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SCATTER_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_SCATTER_HPP_INCLUDED
