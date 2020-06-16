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

#ifndef LBANN_LAYERS_LEARNING_FULLY_CONNECTED_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_FULLY_CONNECTED_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include <layers.pb.h>

#include <string>
#include <sstream>

namespace lbann {

/** @brief Affine transformation
 *
 *  Flattens the input tensor, multiplies with a weights matrix, and
 *  optionally applies an entry-wise bias. Following the
 *  column-vector convention:
 *    @f[ y = W * \text{vec}(x) + b @f]
 *
 *  Two weights are required if bias is applied: the linearity and the
 *  bias. Only the linearity weights are required if bias is not
 *  applied. If weights aren't provided, the linearity weights are
 *  initialized with He normal initialization and the bias weights are
 *  initialized to zero.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class fully_connected_layer : public learning_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The concrete optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  ///@}

public:

  /** @todo Accept a vector for output_size */
  fully_connected_layer(lbann_comm *comm,
                        int output_size,
                        bool transpose = false,
                        WeightsType* weight = nullptr,
                        bool has_bias = true)
    : learning_layer<TensorDataType>(comm),
      m_bias_gradient(nullptr),
      m_transpose(transpose) {

    // Initialize output tensor dimensions
    this->set_output_dims({output_size});

    // Initialize bias
    m_bias_scaling_factor = has_bias ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();

  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning_layer<TensorDataType>(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor),
    m_transpose(other.m_transpose) {

    // Deep matrix copies
    m_bias_gradient = other.m_bias_gradient;
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning_layer<TensorDataType>::operator=(other);
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    m_transpose = other.m_transpose;

    // Deep matrix copies
    deallocate_matrices();
    m_bias_gradient = other.m_bias_gradient;
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

    return *this;
  }

  ~fully_connected_layer() override {
    deallocate_matrices();
  }

  fully_connected_layer* copy() const override {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = learning_layer<TensorDataType>::get_description();
    const auto& bias_str = (m_bias_scaling_factor == El::TypeTraits<TensorDataType>::Zero() ?
                            "disabled" : "enabled");
    desc.add("Bias", bias_str);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override;

  void setup_data(size_t max_mini_batch_size) override {
    learning_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Initialize default weights if none are provided
    if (this->num_weights() > 2) {
      LBANN_ERROR("attempted to setup ", this->get_name(), " with an invalid number of weights");
    }
    if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
      this->set_num_data_type_weights(2);
    } else {
      this->set_num_data_type_weights(1);
    }
    if (!this->has_data_type_weights(0)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto init = make_unique<he_initializer<TensorDataType>>(probability_distribution::gaussian);
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_linearity_weights");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->set_data_type_weights(0, w.get());
      this->m_model->add_weights(std::move(w));
    }
    auto& linearity_weights = this->get_data_type_weights(0);

    // Initialize variance scaling initialization
    auto* cast_initializer
      = dynamic_cast<variance_scaling_initializer<TensorDataType>*>(linearity_weights.get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(this->get_input_size());
      cast_initializer->set_fan_out(this->get_output_size());
    }

    // Setup linearity weights
    auto linearity_dist = this->get_prev_activations().DistData();
    if (linearity_dist.colDist != El::MC
        || linearity_dist.rowDist != El::MR) {
      linearity_dist.colDist = El::STAR;
      linearity_dist.rowDist = El::STAR;
    }
    if (m_transpose) {
      linearity_weights.set_dims(this->get_input_dims(), this->get_output_dims());
    } else {
      linearity_weights.set_dims(this->get_output_dims(), this->get_input_dims());
    }
    linearity_weights.set_matrix_distribution(linearity_dist);

    // Set up bias if needed.
    if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
      if (!this->has_data_type_weights(1)) {
        auto w = make_unique<WeightsType>(this->get_comm());
        auto opt = this->m_model->template create_optimizer<TensorDataType>();
        w->set_name(this->get_name() + "_bias_weights");
        w->set_optimizer(std::move(opt));
        this->set_data_type_weights(1, w.get());
        this->m_model->add_weights(std::move(w));
      }
      auto& bias_weights = this->get_data_type_weights(1);
      // Setup bias weights
      auto bias_dist = this->get_activations().DistData();
      bias_dist.rowDist = El::STAR;
      bias_weights.set_dims(this->get_output_dims());
      bias_weights.set_matrix_distribution(bias_dist);
      if (this->m_bias_gradient != nullptr) {
        El::Zeros(*this->m_bias_gradient,
                  bias_weights.get_matrix_height(),
                  bias_weights.get_matrix_width());
      }
    }

    // Initialize freeze state
    for (auto&& w : this->get_data_type_weights()) {
      if (this->m_frozen) {
        w->freeze();
      } else {
        w->unfreeze();
      }
    }
    for (auto&& w : this->get_data_type_weights()) {
      if (w->is_frozen() != this->is_frozen()) {
        LBANN_ERROR((this->is_frozen() ? "" : "un"), "frozen ",
                    "layer \"", this->get_name(), "\" has ",
                    (w->is_frozen() ? "" : "un"), "frozen ",
                    "weights \"", w->get_name(), "\"");
      }
    }

  }

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Scaling factor for bias term.
   *  If the scaling factor is zero, bias is not applied.
   */
  TensorDataType m_bias_scaling_factor;

  /** Bias weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  AbsDistMatrixType* m_bias_gradient;

  /** Whether the transpose of the linearity matrix is applied. */
  bool m_transpose;

  /** Deallocate distributed matrices. */
  void deallocate_matrices() {
    if (m_bias_gradient != nullptr) delete m_bias_gradient;
  }

  template <typename U>
  friend void fp_compute_impl(fully_connected_layer<U, T_layout, Dev>& l);
  template <typename U>
  friend void bp_compute_impl(fully_connected_layer<U, T_layout, Dev>& l);
};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(fully_connected);

#ifndef LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \
  extern template class fully_connected_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class fully_connected_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_FULLY_CONNECTED_HPP_INCLUDED
