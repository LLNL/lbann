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
class fully_connected_layer : public data_type_layer<TensorDataType> {
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
  fully_connected_layer(
    int output_size,
    bool transpose = false,
    WeightsType* weight = nullptr,
    bool has_bias = true);

  fully_connected_layer(const fully_connected_layer& other);

  fully_connected_layer& operator=(const fully_connected_layer& other);

  ~fully_connected_layer() override;

  fully_connected_layer* copy() const override {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif //LBANN_HAS_ONNX

  description get_description() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:

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
  void deallocate_matrices() {
    if (m_bias_gradient != nullptr) delete m_bias_gradient;
  }

  template <typename U>
  friend void fp_compute_impl(fully_connected_layer<U, T_layout, Dev>& l);
  template <typename U>
  friend void bp_compute_impl(fully_connected_layer<U, T_layout, Dev>& l);
};

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void fully_connected_layer<T, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const {
  //def fully_connected(x, weights=[linearity, bias]):
  //x = Reshape(data=x, shape=[0,-1,1])
  // FIXME: Shape? How can this be different for each reshape node?
  auto* shape = graph.add_value_info();
  shape->set_name(this->get_name() + "_shape_0");
  shape->mutable_type()->mutable_tensor_type()->set_elem_type(1);
  for (auto const& dim : this->get_output_dims())
    shape->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  shape->set_doc_string(this->get_name() + " shape");

  auto const parents = this->get_parent_layers();
  size_t idx = parents[0]->find_child_layer_index(*this);
  auto* reshape = graph.add_node();
  reshape->add_input(parents[0]->get_name() + std::to_string(idx));
  reshape->add_input(this->get_name() + "_shape_0");
  reshape->add_output(this->get_name() + "_reshape_0");
  reshape->set_name(this->get_name() + "_reshape");
  reshape->set_op_type("Reshape");
  reshape->set_domain("");
  reshape->set_doc_string("Reshape node for Fully Connected Layer");

  //linearity = Reshape(data=linearity, shape=[1,linearity_height,linearity_height])
  auto* linearity = graph.add_node();
  linearity->add_input("FIXME: weights[linearity]");
  linearity->add_input("FIXME: shape?");
  linearity->add_output(this->get_name() + "reshape_1");
  linearity->set_name(this->get_name() + "_reshape_1");
  linearity->set_op_type("Reshape");
  linearity->set_domain("");
  linearity->set_doc_string("Reshape (linearity) node for Fully Connected Layer");

  //bias = Reshape(data=bias, shape=[1,-1,1])
  auto* bias = graph.add_node();
  bias->add_input("FIXME: weights[bias]");
  bias->add_input("FIXME: shape?");
  bias->add_output(this->get_name() + "reshape_2");
  bias->set_name(this->get_name() + "_reshape_2");
  bias->set_op_type("Reshape");
  bias->set_domain("");
  bias->set_doc_string("Reshape (bias) node for Fully Connected Layer");

  //z = MatMul(A=linearity, B=x)
  auto* matmul = graph.add_node();
  matmul->add_input(linearity->output(0));
  matmul->add_input(reshape->output(0));
  matmul->add_output(this->get_name() + "matmul_0");
  matmul->set_name(this->get_name() + "matmul");
  matmul->set_op_type("MatMul");
  matmul->set_domain("");
  matmul->set_doc_string("MatMul node for Fully Connected Layer");

  //z = Add(A=z, B=bias)
  auto* add = graph.add_node();
  add->add_input(matmul->output(0));
  add->add_input(bias->output(0));
  add->add_output(this->get_name() + "add_0");
  add->set_name(this->get_name() + "add");
  add->set_op_type("Add");
  add->set_domain("");
  add->set_doc_string("Add node for Fully Connected Layer");

  //z = Reshape(data=z, shape=[0,-1])
  reshape = graph.add_node();
  reshape->add_input(add->output(0));
  reshape->add_input("FIXME: Shape?");
  for (auto const* child : this->get_child_layers()) {
    auto idx = this->find_child_layer_index(*child);
    reshape->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  reshape->set_name(this->get_name() + "_reshape_3");
  reshape->set_op_type("Reshape");
  reshape->set_domain("");
  reshape->set_doc_string("Reshape node for Fully Connected Layer");
  //return z



}
#endif // LBANN_HAS_ONNX

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
