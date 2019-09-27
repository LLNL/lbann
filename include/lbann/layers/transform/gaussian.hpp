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

#ifndef LBANN_LAYER_GAUSSIAN_HPP_INCLUDED
#define LBANN_LAYER_GAUSSIAN_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random values with Gaussian distribution.
 *
 *  During validation and testing, outputs are all equal to the
 *  distribution mean.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class gaussian_layer : public transform_layer<TensorDataType> {
private:
  /** Gaussian distribution mean. */
  TensorDataType m_mean;
  /** Gaussian distribution standard deviation. */
  TensorDataType m_stdev;

public:
  gaussian_layer(lbann_comm *comm,
                 const std::vector<int>& dims,
                 TensorDataType mean = TensorDataType(0),
                 TensorDataType stdev = TensorDataType(1))
    : transform_layer<TensorDataType>(comm), m_mean(mean), m_stdev(stdev) {
    this->set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }
  gaussian_layer* copy() const override { return new gaussian_layer(*this); }
  std::string get_type() const override { return "Gaussian"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer<TensorDataType>::get_description();
    desc.add("Mean", m_mean);
    desc.add("Standard deviation", m_stdev);
    return desc;
  }

protected:

  void fp_compute() override {
    auto& output = this->get_activations();
    if (this->m_model->get_execution_context().get_execution_mode() == execution_mode::training) {
      gaussian_fill(output, output.Height(), output.Width(), m_mean, m_stdev);
    } else {
      El::Fill(output, m_mean);
    }
  }

};

#ifndef LBANN_GAUSSIAN_LAYER_INSTANTIATE
extern template class gaussian_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class gaussian_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class gaussian_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class gaussian_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_GAUSSIAN_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_GAUSSIAN_HPP_INCLUDED
