////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  relu_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~relu_distconv_adapter() = default;

  void setup_distributions(std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents,
                           std::set<dc::Dist*> &updated,
                           std::set<dc::Dist*> &invariants) override;
  void setup_layer(size_t workspace_capacity) override;

  std::unique_ptr<dc::ReLU> m_relu;
};
#endif // LBANN_HAS_DISTCONV

/** Rectified linear unit activation function layer.
 *  \f[ ReLU(x) = \text{max}(x, 0) \f]
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_layer : public data_type_layer<TensorDataType> {
public:
  relu_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {}
  relu_layer* copy() const override { return new relu_layer(*this); }
  std::string get_type() const override { return "ReLU"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:
  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
 protected:
  bool is_distconv_supported() const override { return true; }

  void setup_distconv_adapter() override {
    this->get_dc() = make_unique<relu_distconv_adapter<
      TensorDataType, T_layout, Dev>>(*this);
  }

  relu_distconv_adapter<TensorDataType, T_layout, Dev>& dc() override;
  const relu_distconv_adapter<TensorDataType, T_layout, Dev>& dc() const override;

  void fp_compute_distconv();
  void bp_compute_distconv();
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
relu_distconv_adapter<TensorDataType, T_layout, Dev>&
relu_layer<TensorDataType, T_layout, Dev>::dc() {
  return const_cast<relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const relu_layer<TensorDataType, T_layout, Dev>&>(*this).dc());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const relu_distconv_adapter<TensorDataType, T_layout, Dev>&
relu_layer<TensorDataType, T_layout, Dev>::dc() const {
  return dynamic_cast<const relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      data_type_layer<TensorDataType>::dc());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void relu_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents,
                    std::set<dc::Dist*> &updated,
                    std::set<dc::Dist*> &invariants) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      equivalents, updated, invariants);

  auto &x = this->get_prev_activations_dist();
  auto &y = this->get_activations_dist();
  auto &dx = this->get_error_signals_dist();
  auto &dy = this->get_prev_error_signals_dist();

  // x == dx
  equivalents[&x].insert(&dx);
  equivalents[&dx].insert(&x);
  // y == dy
  equivalents[&y].insert(&dy);
  equivalents[&dy].insert(&y);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void relu_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  m_relu = make_unique<dc::ReLU>(dc::get_backend());
  m_relu->setup(this->get_prev_activations(),
                this->get_activations(),
                this->get_error_signals(),
                this->get_prev_error_signals());
}
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
