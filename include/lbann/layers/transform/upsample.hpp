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

#ifndef LBANN_LAYER_UPSAMPLE_HPP_INCLUDED
#define LBANN_LAYER_UPSAMPLE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/dnn_enums.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/dnn_lib/upsample.hpp"
#endif // LBANN_HAS_DNN_LIB
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

#include <utility>
#include <vector>

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

enum class upsample_mode{
  NEAREST
};

inline upsample_mode to_upsample_mode(std::string m)
{
  if (m == "nearest")
    return upsample_mode::NEAREST;
  else {
    LBANN_ERROR("Invalid upsample mode requested.");
  }
}

#ifdef LBANN_HAS_DISTCONV

namespace dc {
using Shape = ::distconv::tensor::Shape;
using Backend = ::distconv::BackendDNNLib;
} // namespace dc

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class upsample_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  upsample_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~upsample_distconv_adapter() = default;
  dc::Shape get_activations_local_shape(int index = 0) const override;
  void setup_layer(size_t workspace_capacity) override;
  void
  fp_compute(bool training = true); // training=true for max back-compatibility.
  void bp_compute();
private:
  dnn_lib::TensorDescriptor m_xdesc;
  dnn_lib::TensorDescriptor m_ydesc;
  dnn_lib::TensorDescriptor m_dxdesc;
  dnn_lib::TensorDescriptor m_dydesc;
};
#endif // LBANN_HAS_DISTCONV

// Forward declaration
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class unupsample_layer;

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class upsample_layer : public data_type_layer<TensorDataType>
{
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "upsample only supports DATA_PARALLEL");

private:
  /** Upsample mode. */
  upsample_mode m_upsample_mode;

  /** Output scale factors. */
  std::vector<int> m_scale_factors;

#ifdef LBANN_HAS_DNN_LIB
  /** Pooling descriptor. */
  dnn_lib::PoolingDescriptor m_pooling_dnn_desc;
  /** Tensor DNN library descriptors. */
  dnn_lib::data_parallel_layer_tensor_manager<TensorDataType>
    m_tensors_dnn_desc;
#endif // LBANN_HAS_DNN_LIB

public:
  upsample_layer(lbann_comm* comm,
                int num_data_dims,
                int scale_factors,
                upsample_mode mode)
    : upsample_layer(comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, scale_factors),
                    mode)
  {}

  upsample_layer(lbann_comm* comm,
                int num_data_dims,
                std::vector<int> scale_factors,
                upsample_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_upsample_mode(mode),
      m_scale_factors(scale_factors)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {}

  upsample_layer(const upsample_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_upsample_mode(other.m_upsample_mode),
      m_scale_factors(other.m_scale_factors)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_pooling_dnn_desc(other.m_pooling_dnn_desc),
      m_tensors_dnn_desc(other.m_tensors_dnn_desc)
#endif // LBANN_HAS_DNN_LIB
  {
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  upsample_layer& operator=(const upsample_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_upsample_mode = other.m_upsample_mode;
    m_scale_factors = other.m_scale_factors;
#ifdef LBANN_HAS_DNN_LIB
    m_pooling_dnn_desc = other.m_pooling_dnn_desc;
    m_tensors_dnn_desc = other.m_tensors_dnn_desc;
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
    return *this;
  }

  ~upsample_layer() override = default;

  upsample_layer* copy() const override { return new upsample_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "upsample"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::stringstream ss;

    // Upsample mode
    ss.str(std::string{});
    ss.clear();
    switch (m_upsample_mode) {
    case upsample_mode::NEAREST:
      ss << "nearest";
      break;
    default:
      ss << "invalid";
    }
    desc.add("Upsample mode", ss.str());

    // Upsample scale factors
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_scale_factors.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_scale_factors[i];
    }
    desc.add("Scale factors", ss.str());

    // Result
    return desc;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  upsample_layer() : upsample_layer(nullptr, 1, 1, upsample_mode::NEAREST) {}

  void setup_dims(DataReaderMetaData& dr_metadata) override
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      output_dims[i + 1] = m_scale_factors[i] * input_dims[i + 1];
    }
    this->set_output_dims(output_dims);
  }

  /// Initialize GPU objects
  void setup_gpu() override
  {
    data_type_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else

    // Set upsample descriptor
    int ndims = m_scale_factors.size();
    std::vector<int> padding(ndims, 0);
    m_pooling_dnn_desc.set(pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING,
                           dnn_lib::DNN_PROPAGATE_NAN,
                           ndims,
                           m_scale_factors.data(),
                           padding.data(),
                           m_scale_factors.data());

#endif // #ifndef LBANN_HAS_DNN_LIB
  }

  void fp_compute() override;

  void bp_compute() override;

private:
  /// Pooling forward propagation with DNN library
  void fp_compute_dnn();

  /// Pooling backward propagation with DNN library
  void bp_compute_dnn();

  /// Pooling forward propagation with im2col
  void fp_compute_im2col();

  /// Pooling forward propagation with im2col
  void bp_compute_im2col();

#ifdef LBANN_HAS_DISTCONV
  friend class upsample_distconv_adapter<TensorDataType, T_layout, Dev>;

protected:
  bool is_distconv_supported() const override;
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override
  {
    this->get_distconv_adapter_ptr() =
      std::make_unique<upsample_distconv_adapter<TensorDataType, T_layout, Dev>>(
        *this);
  }
  upsample_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() override;
  const upsample_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_UPSAMPLE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class upsample_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_UPSAMPLE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_UPSAMPLE_HPP_INCLUDED
