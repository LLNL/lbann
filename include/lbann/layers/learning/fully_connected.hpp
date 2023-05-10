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

#ifndef LBANN_LAYERS_LEARNING_FULLY_CONNECTED_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_FULLY_CONNECTED_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"

#include <string>

namespace lbann {

/** @brief Affine transformation
 *
 *  Flattens the input tensor, multiplies with a weights matrix, and
 *  optionally applies an entry-wise bias. Following a row-vector
 *  convention:
 *    @f[ y = \text{vec}(x) W^T + b @f]
 *
 *  Two weights are required if bias is applied: the linearity and the
 *  bias. Only the linearity weights are required if bias is not
 *  applied. If weights aren't provided, the linearity weights are
 *  initialized with He normal initialization and the bias weights are
 *  initialized to zero.
 *
 *  For flat data, this layer is similar to Keras' dense layer or
 *  PyTorch's linear operation. However, it implicitly flattens
 *  multi-dimensional data. To avoid this flattening, consider the
 *  channel-wise fully-connected layer.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class fully_connected_layer : public data_type_layer<TensorDataType>
{
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
  fully_connected_layer(int output_size,
                        bool transpose = false,
                        WeightsType* weight = nullptr,
                        bool has_bias = true);

  fully_connected_layer(const fully_connected_layer& other);

  fully_connected_layer& operator=(const fully_connected_layer& other);

  ~fully_connected_layer() override;

  fully_connected_layer* copy() const override
  {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS | PREV_ACTIVATIONS;
  }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  description get_description() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  fully_connected_layer();

  void setup_data(size_t max_mini_batch_size) override;

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
  void deallocate_matrices()
  {
    if (m_bias_gradient != nullptr)
      delete m_bias_gradient;
  }

  template <typename U>
  friend void fp_compute_impl(fully_connected_layer<U, T_layout, Dev>& l);
  template <typename U>
  friend void bp_compute_impl(fully_connected_layer<U, T_layout, Dev>& l);
};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(fully_connected);

#ifndef LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class fully_connected_layer<T,                               \
                                              data_layout::DATA_PARALLEL,      \
                                              Device>;                         \
  extern template class fully_connected_layer<T,                               \
                                              data_layout::MODEL_PARALLEL,     \
                                              Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_FULLY_CONNECTED_HPP_INCLUDED
