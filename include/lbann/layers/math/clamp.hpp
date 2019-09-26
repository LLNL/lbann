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

#ifndef LBANN_LAYERS_MATH_CLAMP_HPP_INCLUDED
#define LBANN_LAYERS_MATH_CLAMP_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Constrain values to a range.
 *
 *  @f[
 *    \text{clamp}(x; \text{min}, \text{max}) =
 *      \begin{cases}
 *        \text{min} & x \leq \text{min}           \\
 *        x          & \text{min} < x < \text{max} \\
 *        \text{max} & x \geq \text{max}
 *      \end{cases}
 *  @f]
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class clamp_layer : public data_type_layer<TensorDataType> {
public:
  clamp_layer(lbann_comm *comm, TensorDataType min, TensorDataType max)
    : data_type_layer<TensorDataType>(comm), m_min(min), m_max(max) {
    if (m_min > m_max) {
      std::stringstream err;
      err << "[" << m_min << "," << m_max << "] is an invalid range";
      LBANN_ERROR(err.str());
    }
  }
  clamp_layer* copy() const override { return new clamp_layer(*this); }
  std::string get_type() const override { return "clamp"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::stringstream ss;
    ss << "[" << m_min << "," << m_max << "]";
    desc.add("Range", ss.str());
    return desc;
  }

protected:
  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    set_output_dims(get_input_dims());
  }
  void fp_compute() override;
  void bp_compute() override;

private:
  /** Minimum output. */
  TensorDataType m_min;
  /** Maximum output. */
  TensorDataType m_max;

};

#ifndef LBANN_CLAMP_LAYER_INSTANTIATE
extern template class clamp_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class clamp_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class clamp_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class clamp_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_CLAMP_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MATH_CLAMP_HPP_INCLUDED
