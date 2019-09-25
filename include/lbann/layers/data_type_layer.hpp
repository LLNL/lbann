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

#ifndef LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

// Forward declarations
template <typename TensorDataType>
class weights;

using supported_layer_data_type = El::TypeList<float/*, double*/>;

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
  static_assert(is_supported_layer_data_type<TensorDataType>::value,
                "Must use a supported type.");

  data_type_layer(lbann_comm *comm) : Layer(comm), m_frozen(false) {}
  data_type_layer(const data_type_layer<TensorDataType>& other);
  data_type_layer& operator=(const data_type_layer<TensorDataType>& other);
  virtual ~data_type_layer() = default;

  // ===========================================================
  // Weights access functions
  // ===========================================================

  /** Get references to weights. */
  inline std::vector<weights<TensorDataType>*>& get_weights() { return m_weights; }
  /** Get references to weights. (const) */
  inline const std::vector<weights<TensorDataType>*>& get_weights() const { return m_weights; }
  /** Set list of pointers to weights. */
  inline void set_weights(std::vector<weights<TensorDataType>*> w) { get_weights() = w; }
  /** Replace weights with another Layer's weights*/
  void replace_weights(data_type_layer<TensorDataType>* other_layer);

  // ===========================================================
  // Tensor dimension access functions
  // ===========================================================

  /** Get input tensor dimensions. */
  std::vector<int> get_input_dims(int input_index = 0) const;
  /** Get input tensor size. */
  int get_input_size(int input_index = 0) const;
  /** Get output tensor dimensions. */
  std::vector<int> get_output_dims(int output_index = 0) const;
  /** Get output tensor size. */
  int get_output_size(int output_index = 0) const;

  /** Set output tensor dimensions. */
  void set_output_dims(std::vector<int> dims, int output_index = 0);

  // ===========================================================
  // Tensor access functions
  // ===========================================================

  /** Get activation tensor. */
  El::AbstractDistMatrix<TensorDataType>& get_activations(int child_index = 0);
  /** Get error signal tensor. */
  El::AbstractDistMatrix<TensorDataType>& get_error_signals(int parent_index = 0);
  /** Get previous activation tensor. */
  const El::AbstractDistMatrix<TensorDataType>& get_prev_activations(int parent_index = 0) const;
  /** Get activation tensor. */
  const El::AbstractDistMatrix<TensorDataType>& get_activations(int child_index = 0) const;
  /** Get previous error signal tensor. */
  const El::AbstractDistMatrix<TensorDataType>& get_prev_error_signals(int child_index = 0) const;
  /** Get error signal tensor. */
  const El::AbstractDistMatrix<TensorDataType>& get_error_signals(int parent_index = 0) const;
  /** Get local portion of activation tensor. */
  El::AbstractMatrix<TensorDataType>& get_local_activations(int child_index = 0);
  /** Get local portion of error signal tensor. */
  El::AbstractMatrix<TensorDataType>& get_local_error_signals(int parent_index = 0);
  /** Get local portion of previous activation tensor. */
  const El::AbstractMatrix<TensorDataType>& get_local_prev_activations(int parent_index = 0) const;
  /** Get local portion of activation tensor. */
  const El::AbstractMatrix<TensorDataType>& get_local_activations(int child_index = 0) const;
  /** Get local portion of previous error signal tensor. */
  const El::AbstractMatrix<TensorDataType>& get_local_prev_error_signals(int child_index = 0) const;
  /** Get local portion of error signal tensor. */
  const El::AbstractMatrix<TensorDataType>& get_local_error_signals(int parent_index = 0) const;

  // ===========================================================
  // Hint layer access functions
  // ===========================================================

  /** Set hint layer.
   *  Properties of the hint layer are used during the setup
   *  phase. For instance, the output tensor dimensions are set to
   *  match the hint layer's first output tensor.
   */
  void set_hint_layer(const data_type_layer* l) { m_hint_layer = l; }

  /** Get hint layer. */
  const data_type_layer* get_hint_layer() const { return m_hint_layer; }

  // ===========================================================
  // Freeze management functions
  // ===========================================================

  void freeze();
  void unfreeze();
  bool is_frozen() const;

protected:

  // ===========================================================
  // Setup helper functions
  // ===========================================================
  /** Setup tensor dimensions
   *  Called by the 'setup' function. If there are any input tensors,
   *  the base method sets all uninitialized output tensor dimensions
   *  equal to the first input tensor dimensions.
   */
  virtual void setup_dims();
  /** Setup distributed matrices.
   *  Called by the 'setup' function. Each column of these distributed
   *  matrices is interpreted as the flattened tensor for a mini-batch
   *  sample. The matrices themselves are constructed by calling the
   *  'construct_matrix' function. If any matrices have already been
   *  setup, they are destroyed and reinstantiated.
   */
  virtual void setup_matrices(const El::Grid& grid);
  /** Construct distributed matrix.
   *  Called by the 'setup_matrices' function. 'type' is one of the
   *  following: "input", "output", "gradient_wrt_output",
   *  "gradient_wrt_input".
   */
  virtual std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> construct_matrix(const El::Grid& grid,
                                                       std::string type,
                                                       El::Int index);

  // ===========================================================
  // Protected class members
  // ===========================================================
  /** References to layer weights. */
  std::vector<weights<TensorDataType>*> m_weights;

  /** Avoid back prop if frozen */
  bool m_frozen;

private:

  // ===========================================================
  // Private access functions
  // ===========================================================

  /** Get activation tensor corresponding to child layer. */
  const El::AbstractDistMatrix<TensorDataType>& get_activations(const data_type_layer& child) const;
  /** Get error signal tensor corresponding to parent layer. */
  const El::AbstractDistMatrix<TensorDataType>& get_error_signals(const data_type_layer& parent) const;

  // ===========================================================
  // Private class members
  // ===========================================================

  /** Dimensions of output tensors. */
  std::vector<std::vector<int>> m_output_dims_list;

  /** Input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<El::AbstractDistMatrix<TensorDataType>>> m_inputs;
  /** Output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<El::AbstractDistMatrix<TensorDataType>>> m_outputs;
  /** Objective function gradients w.r.t. the output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<El::AbstractDistMatrix<TensorDataType>>> m_gradient_wrt_outputs;
  /** Objective function gradients w.r.t. the input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<El::AbstractDistMatrix<TensorDataType>>> m_gradient_wrt_inputs;

  /** Hint layer.
   *  During setup, the output tensor dimensions are set to match the
   *  first output tensor of the hint layer. Derived classes may do
   *  more elaborate setup based on the hint layer.
   */
  const data_type_layer* m_hint_layer = nullptr;
};

} // namespace lbann

#include "data_type_layer_impl.hpp"

#endif // LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
