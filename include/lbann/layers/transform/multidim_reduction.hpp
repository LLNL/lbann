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

#ifndef LBANN_LAYER_MULTIDIM_REDUCTION_HPP_INCLUDED
#define LBANN_LAYER_MULTIDIM_REDUCTION_HPP_INCLUDED

#include <sstream>
#include <unordered_map>
#include <vector>

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

enum class multidim_reduction_mode
{
  INVALID,
  SUM,
  PRODUCT,
  MAX,
  MIN,
};

/** @brief Reduce tensor to scalar
 *
 *  @todo Reduction over specified dimensions.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class multidim_reduction_layer : public data_type_layer<TensorDataType>
{
private:
  /** Reduction dimensions. */
  std::vector<unsigned int> m_axes;

  /** Reduction mode. */
  multidim_reduction_mode m_mode;

  /** Reduced dimensions */
  std::vector<int32_t> m_input_modes, m_output_modes;

public:
  multidim_reduction_layer(
    const std::vector<unsigned int>& axes = {},
    multidim_reduction_mode mode = multidim_reduction_mode::SUM);

  multidim_reduction_layer* copy() const override
  {
    return new multidim_reduction_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "multidim reduction"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::string mode_str;
    switch (m_mode) {
    case multidim_reduction_mode::SUM:
      mode_str = "sum";
      break;
    case multidim_reduction_mode::PRODUCT:
      mode_str = "product";
      break;
    case multidim_reduction_mode::MAX:
      mode_str = "max";
      break;
    case multidim_reduction_mode::MIN:
      mode_str = "min";
      break;
    case multidim_reduction_mode::INVALID:
    default:
      mode_str = "invalid";
    }
    desc.add("Mode", mode_str);

    std::stringstream axes_str;
    for (size_t i = 0; i < m_axes.size(); ++i) {
      axes_str << m_axes[i];
      if (i != m_axes.size() - 1) {
        axes_str << ", ";
      }
    }
    desc.add("Axes", axes_str.str());

    return desc;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims() override;

  void fp_compute() override;

  void bp_compute() override;
};

#ifndef LBANN_MULTIDIM_REDUCTION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class multidim_reduction_layer<T,                            \
                                                 data_layout::DATA_PARALLEL,   \
                                                 Device>;                      \
  extern template class multidim_reduction_layer<T,                            \
                                                 data_layout::MODEL_PARALLEL,  \
                                                 Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_MULTIDIM_REDUCTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REDUCTION_HPP_INCLUDED
