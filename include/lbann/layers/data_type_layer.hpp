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

#ifndef LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/weights/data_type_weights.hpp"

namespace lbann {

// Forward declarations
//template <typename TensorDataType>
//class data_type_weights;

using supported_layer_data_type = El::TypeList<DataType, float/*, double*/>;

template <typename T, typename List> struct IsElement;

template <typename Head, typename... Tail>
struct IsElement<Head, El::TypeList<Head,Tail...>> : std::true_type {};

template <typename T, typename Head, typename... Tail>
struct IsElement<T, El::TypeList<Head,Tail...>> : IsElement<T, El::TypeList<Tail...>> {};

template <typename T>
struct IsElement<T, El::TypeList<>> : std::false_type {};

template <typename T> using is_supported_layer_data_type = IsElement<T, supported_layer_data_type>;

template <typename TensorDataType>
class data_type_layer : public Layer {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The local tensor type expected in this object. */
  using AbsMatrixType = El::AbstractMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  static_assert(is_supported_layer_data_type<TensorDataType>::value,
                "Must use a supported type.");

  data_type_layer(lbann_comm *comm) : Layer(comm) {}
  data_type_layer(const data_type_layer<TensorDataType>& other);
  data_type_layer& operator=(const data_type_layer<TensorDataType>& other);
  virtual ~data_type_layer() = default;

  /** Forward propagation step.
   *  Apply a mathematical operation to input tensors to obtain output
   *  tensors.
   */
  void forward_prop() override;
  /** Backward propagation step.
   *  Given the objective function gradients w.r.t. the output
   *  tensors, compute the gradients w.r.t. the input tensors and
   *  w.r.t. the weights. This is essentially an application of the
   *  chain rule.
   */
  void back_prop() override;

  void summarize_matrices(lbann_summary& summarizer, int step) override;

  /** Check that the setup is reasonable. */
  void check_setup() override;

  // ===========================================================
  // Weights access functions
  // ===========================================================

  /** @brief Set list of pointers to weights. */
  void set_weights(std::vector<weights*>& w) override {
    m_weights.resize(w.size());
    std::transform(begin(w), end(w), begin(m_weights),
                   [](weights* wptr) {
                     return (wptr
                             ? &(dynamic_cast<WeightsType&>(*wptr))
                             : nullptr);
                   });
  }

  /** @brief Replace weights with another Layer's weights*/
  void replace_weights(Layer* other_layer) override;

  // ===========================================================
  // Tensor access functions
  // ===========================================================

  /** Get activation tensor. */
  AbsDistMatrixType& get_activations(int child_index = 0);
  /** Get error signal tensor. */
  AbsDistMatrixType& get_error_signals(int parent_index = 0);
  /** Get previous activation tensor. */
  const AbsDistMatrixType& get_prev_activations(int parent_index = 0) const;
  /** Get activation tensor. */
  const AbsDistMatrixType& get_activations(int child_index = 0) const;
  /** Get previous error signal tensor. */
  const AbsDistMatrixType& get_prev_error_signals(int child_index = 0) const;
  /** Get error signal tensor. */
  const AbsDistMatrixType& get_error_signals(int parent_index = 0) const;
  /** Get local portion of activation tensor. */
  AbsMatrixType& get_local_activations(int child_index = 0);
  /** Get local portion of error signal tensor. */
  AbsMatrixType& get_local_error_signals(int parent_index = 0);
  /** Get local portion of previous activation tensor. */
  const AbsMatrixType& get_local_prev_activations(int parent_index = 0) const;
  /** Get local portion of activation tensor. */
  const AbsMatrixType& get_local_activations(int child_index = 0) const;
  /** Get local portion of previous error signal tensor. */
  const AbsMatrixType& get_local_prev_error_signals(int child_index = 0) const;
  /** Get local portion of error signal tensor. */
  const AbsMatrixType& get_local_error_signals(int parent_index = 0) const;

protected:

  // ===========================================================
  // Setup helper functions
  // ===========================================================

  /** Setup distributed matrices.
   *  Called by the 'setup' function. Each column of these distributed
   *  matrices is interpreted as the flattened tensor for a mini-batch
   *  sample. The matrices themselves are constructed by calling the
   *  'construct_matrix' function. If any matrices have already been
   *  setup, they are destroyed and reinstantiated.
   */
  void setup_matrices(const El::Grid& grid) override;
  /** Construct distributed matrix.
   *  Called by the 'setup_matrices' function. 'type' is one of the
   *  following: "input", "output", "gradient_wrt_output",
   *  "gradient_wrt_input".
   */
  virtual std::unique_ptr<AbsDistMatrixType> construct_matrix(const El::Grid& grid,
                                                       std::string type,
                                                       El::Int index);
  /** Setup layer data.
   *  Called by the 'setup' function. Memory is allocated for
   *  distributed matrices.
   */
  void setup_data() override;

  // ===========================================================
  // Forward prop step helper functions
  // ===========================================================

  /** Setup input tensors.
   *  Called by the 'forward_prop' function. Each input tensor is
   *  setup as a view or copy of the corresponding parent layer's
   *  output tensor.
   */
  void fp_setup_inputs(El::Int mini_batch_size) override;
  /** Setup output tensors.
   *  Called by the 'forward_prop' function. Each output tensor is
   *  resized to match the mini-batch size.
   */
  void fp_setup_outputs(El::Int mini_batch_size) override;

  // ===========================================================
  // Back prop step helper functions
  // ===========================================================

  /** Setup gradient w.r.t. output tensors.
   *  Called by the 'back_prop' function. Each gradient w.r.t. output
   *  tensor is setup as a view or copy of the corresponding child
   *  layer's gradient w.r.t. input tensor.
   */
  void bp_setup_gradient_wrt_outputs(El::Int mini_batch_size) override;
  /** Setup gradient w.r.t. input tensors.
   *  Called by the 'back_prop' function. Each gradient w.r.t. input
   *  tensor is resized to match the mini-batch size.
   */
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;
  /** Compute objective funciton gradients.
   *  Called by the 'back_prop' function. Given the input, output, and
   *  gradient w.r.t. output tensors, the gradient w.r.t. input
   *  tensors are populated with the computed values and the gradients
   *  w.r.t. the weights are sent to the appropriate optimizers.
   */
  void bp_compute() override;

  // ===========================================================
  // Protected Weights access functions
  // ===========================================================

  /** Get references to weights. */
  std::vector<WeightsType*>& get_data_type_weights() { return m_weights; }
  /** Get references to weights. (const) */
  const std::vector<WeightsType*>& get_data_type_weights() const {
    return m_weights;
  }

  /** @brief Get a specific weights object */
  WeightsType& get_data_type_weights(size_t idx) {
    return *(m_weights.at(idx));
  }
  WeightsType const& get_data_type_weights(size_t idx) const {
    return *(m_weights.at(idx));
  }

  bool has_data_type_weights(size_t idx) const noexcept {
    return (idx < m_weights.size() && m_weights[idx] != nullptr);
  }

  void set_num_data_type_weights(size_t num_weights) {
    m_weights.resize(num_weights, nullptr);
  }

  void set_data_type_weights(size_t idx, WeightsType* w) {
    m_weights.at(idx) = w;
  }

  /** Set list of pointers to weights. */
  void set_data_type_weights(std::vector<WeightsType*> w) { m_weights = w; }
  /** Replace weights with another Layer's weights*/
  //void replace_weights(Layer* other_layer) override;

  void add_weights(WeightsType* w) { m_weights.push_back(w); }
  size_t num_weights() const noexcept { return m_weights.size(); }
  bool has_weights() const noexcept { return num_weights() > 0; }

private:
  // ===========================================================
  // Private access functions
  // ===========================================================

  /** @brief Get references to weights. */
  std::vector<weights*> get_weights() override {
    return std::vector<weights*>(begin(m_weights), end(m_weights));
  }

  /** @brief Get references to weights. (const) */
  std::vector<weights const*> get_weights() const override {
    return std::vector<weights const*>(begin(m_weights), end(m_weights));
  }

  /** Get activation tensor corresponding to child layer. */
  const AbsDistMatrixType& get_activations(const data_type_layer& child) const;
  /** Get error signal tensor corresponding to parent layer. */
  const AbsDistMatrixType& get_error_signals(const data_type_layer& parent) const;

  // ===========================================================
  // Private class members
  // ===========================================================

  /** References to layer weights. */
  std::vector<WeightsType*> m_weights;

  /** Input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMatrixType>> m_inputs;
  /** Output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMatrixType>> m_outputs;
  /** Objective function gradients w.r.t. the output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMatrixType>> m_gradient_wrt_outputs;
  /** Objective function gradients w.r.t. the input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMatrixType>> m_gradient_wrt_inputs;

};

#ifndef LBANN_DATA_TYPE_LAYER_INSTANTIATE
extern template class data_type_layer<DataType>;
#endif // LBANN_DATA_TYPE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
