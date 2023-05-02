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

#ifndef LBANN_LAYER_UNIFORM_HPP_INCLUDED
#define LBANN_LAYER_UNIFORM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random tensor with uniform distribution */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class uniform_layer : public data_type_layer<TensorDataType>
{
private:
  /** @brief Uniform distribution minimum. */
  TensorDataType m_min;
  /** @brief Uniform distribution maximum. */
  TensorDataType m_max;
  /** @brief Whether to have deterministic output when not training.
   *
   *  If true, the tensor is filled with the distribution mean during
   *  evaluation.
   */
  bool m_training_only;

public:
  uniform_layer(lbann_comm* comm,
                std::vector<int> dims,
                TensorDataType min = El::TypeTraits<TensorDataType>::Zero(),
                TensorDataType max = El::TypeTraits<TensorDataType>::One(),
                bool training_only = false)
    : data_type_layer<TensorDataType>(comm),
      m_min(min),
      m_max(max),
      m_training_only(training_only)
  {
    this->set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }
  uniform_layer* copy() const override { return new uniform_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "uniform"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::stringstream ss;
    ss << "[" << m_min << "," << m_max << ")";
    desc.add("Range", ss.str());
    desc.add("Training only", m_training_only);
    return desc;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  uniform_layer() : uniform_layer(nullptr, {1}) {}

  void fp_compute() override;
};

#ifndef LBANN_UNIFORM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class uniform_layer<T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class uniform_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_UNIFORM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_UNIFORM_HPP_INCLUDED
