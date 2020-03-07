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

#include <set>
#include <map>
#include <array>

namespace lbann {

// Forward declarations
namespace cudnn {
template <typename U>
class data_parallel_layer_tensor_manager;
template <typename U>
class entrywise_layer_tensor_manager;
}

using supported_layer_data_type = El::TypeList<float, double>;

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

  /** @brief The proxy tensor type expected in this object. */
  template <El::Device D>
  using AbsDistMatReadProxyType = El::AbstractDistMatrixReadDeviceProxy<TensorDataType, D>;

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

  /** Get a string representing the layer datatype
   */
  std::string get_datatype_name() const override {
    return TypeName<TensorDataType>();
  };

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
  // Public Tensor access functions
  // ===========================================================

  /** Get activation tensor corresponding to child layer. */
  const BaseDistMat& get_activations(const Layer& child) const override;
  /** Get error signal tensor corresponding to parent layer. */
  const BaseDistMat& get_error_signals(const Layer& parent) const override;

  /** Get activation tensor. */
  AbsDistMatrixType& get_activations(int child_index = 0);
  /** Get error signal tensor. */
  AbsDistMatrixType& get_error_signals(int parent_index = 0);
  /** Get activation tensor. */
  const AbsDistMatrixType& get_activations(int child_index = 0) const;
  /** Get error signal tensor. */
  const AbsDistMatrixType& get_error_signals(int parent_index = 0) const;

  /** Get local portion of activation tensor. */
  AbsMatrixType& get_local_activations(int child_index = 0);
  /** Get local portion of error signal tensor. */
  AbsMatrixType& get_local_error_signals(int parent_index = 0);
  /** Get local portion of activation tensor. */
  const AbsMatrixType& get_local_activations(int child_index = 0) const;
  /** Get local portion of error signal tensor. */
  const AbsMatrixType& get_local_error_signals(int parent_index = 0) const;

protected:

  // ===========================================================
  // Protected Tensor access functions
  // ===========================================================

  /** Get previous activation tensor. */
  const AbsDistMatrixType& get_prev_activations(int parent_index = 0) const;
  /** Get previous error signal tensor. */
  const AbsDistMatrixType& get_prev_error_signals(int child_index = 0) const;

  /** Get local portion of previous activation tensor. */
  const AbsMatrixType& get_local_prev_activations(int parent_index = 0) const;
  /** Get local portion of previous error signal tensor. */
  const AbsMatrixType& get_local_prev_error_signals(int child_index = 0) const;

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

#ifdef LBANN_HAS_DISTCONV
 public:

  using TensorDevType = dc::TensorDev<TensorDataType>;
  using TensorShufflerType = dc::TensorShuffler<TensorDataType>;

  void init_distribution(
      std::map<const Layer*, std::array<lbann::dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override;
  void setup_tensor_distribution_add_adjacent_invariants(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants) override;

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override {}
  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override {}
  void setup_distconv_post(size_t ws_size) override {}

  virtual const TensorDevType &get_prev_activations_t() const;
  virtual TensorDevType &get_prev_activations_t();
  virtual const TensorDevType &get_prev_activations_const_view() const;
  virtual const TensorDevType &get_activations_t() const;
  virtual TensorDevType &get_activations_t();
  virtual const TensorDevType &get_activations_t(const Layer &child) const;
  virtual TensorDevType &get_activations_copyout();

  virtual const TensorDevType &get_prev_error_signals_t() const;
  virtual TensorDevType &get_prev_error_signals_t();
  virtual const TensorDevType &get_prev_error_signals_const_view() const;
  virtual const TensorDevType &get_error_signals_t() const;
  virtual const TensorDevType &get_error_signals_t(const Layer &parent) const;
  virtual TensorDevType &get_error_signals_t();
  virtual TensorDevType &get_error_signals_copyout();

 protected:
  bool using_distconv() const override;
  void setup_distconv() override;

  virtual int get_num_dims() const;
  virtual int get_num_spatial_dims() const;
  /** Return Distconv-related shapes. */
  const dc::Shape get_input_tensor_shape() const;
  const dc::Shape get_output_tensor_shape(int output_index = 0) const;

  // Copis and converts input or output tensors when necessary
  void ensure_prev_activations();
  void copy_out_activations();
  void ensure_prev_error_signals();
  void copy_out_error_signals();

  bool parent_copy_in_required(size_t input_index) const;
  bool parent_shuffle_required(size_t input_index) const;
  bool child_copy_out_required(size_t output_index) const;
  bool child_shuffle_required(size_t output_index) const;
  bool keep_original_input(size_t input_index) const;
  bool keep_original_output(size_t output_index) const;
  bool keep_original() const;

  /** Initialize distconv tensors */
  virtual void setup_prev_activations_tensor(const std::array<dc::Dist, dc::num_dists> &dists);
  virtual dc::Shape get_activations_tensor_local_shape() const;
  virtual void setup_activations_tensor(const std::array<dc::Dist, dc::num_dists> &dists,
                                        bool allocate=true);
  virtual void setup_activations_copyout_tensor(const std::array<dc::Dist, dc::num_dists> &dists);

  virtual void setup_prev_error_signals_tensor(const std::array<dc::Dist, dc::num_dists> &dists);
  virtual void setup_error_signals_tensor(const std::array<dc::Dist, dc::num_dists> &dists);
  virtual void setup_error_signals_copyout_tensor(const std::array<dc::Dist, dc::num_dists> &dists);

  virtual void fp_setup_distconv(El::Int mini_batch_size) override;
  virtual void bp_setup_distconv(El::Int mini_batch_size) override;

  virtual size_t estimate_memory_usage(const std::array<dc::Dist, dc::num_dists> &dists);

  void dump_activations() const;
  void dump_reference_activations();
  void dump_error_signals() const;
  void dump_reference_error_signals();

 private:
  /** Previous activation tensor */
  TensorDevType m_prev_activations_t;
  /** View to Elemental matrix of previous activations */
  TensorDevType m_prev_activations_const_view;
  /** Activation tensor */
  TensorDevType m_activations_t;
  /** Elemental-format activation matrix */
  TensorDevType m_activations_copyout;
  /** Previous error signal tensor */
  TensorDevType m_prev_error_signals_t;
  /** View to Elemental matrix */
  TensorDevType m_prev_error_signals_const_view;
  /** Error signal tensor */
  TensorDevType m_error_signals_t;
  /** Elemental-format matrix */
  TensorDevType m_error_signals_copyout;
  std::vector<bool> m_parent_copy_in_required;
  std::vector<bool> m_parent_shuffle_required;
  std::vector<bool> m_child_copy_out_required;
  std::vector<bool> m_child_shuffle_required;
  TensorShufflerType *m_prev_activations_shuffler = nullptr;
  TensorShufflerType *m_prev_activations_shuffler_last_mb[3];
  TensorShufflerType *m_activations_shuffler = nullptr;
  TensorShufflerType *m_activations_shuffler_last_mb[3];
  TensorShufflerType *m_prev_error_signals_shuffler = nullptr;
  TensorShufflerType *m_prev_error_signals_shuffler_last_mb[3];
  TensorShufflerType *m_error_signals_shuffler = nullptr;
  TensorShufflerType *m_error_signals_shuffler_last_mb[3];
  std::vector<bool> m_keep_original_input;
  std::vector<bool> m_keep_original_output;

  void setup_keep_original_tensors();
  void setup_inter_layer_adaptation();
#endif // LBANN_HAS_DISTCONV

#ifdef LBANN_HAS_CUDA
  template <typename U>
  friend class cudnn::data_parallel_layer_tensor_manager;
  template <typename U>
  friend class cudnn::entrywise_layer_tensor_manager;
#endif // LBANN_HAS_CUDA
};

#ifndef LBANN_DATA_TYPE_LAYER_INSTANTIATE
#define PROTO(T)                           \
  extern template class data_type_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_DATA_TYPE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
