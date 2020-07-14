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

#include "lbann/utils/h2_tmp.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#include <set>
#include <map>
#include <array>
#endif // LBANN_HAS_DISTCONV

namespace lbann {

// Forward declarations
namespace cudnn {
template <typename U>
class data_parallel_layer_tensor_manager;
template <typename U>
class entrywise_layer_tensor_manager;
}

using supported_layer_data_type = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  fp16,
#endif
#ifdef LBANN_HAS_HALF
  cpu_fp16,
#endif
  float, double>;

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
  static_assert(
    h2::meta::tlist::MemberV<TensorDataType, supported_layer_data_type>(),
    "Must use a supported type.");

  data_type_layer(lbann_comm *comm, bool persistent_error_signals=false)
    : Layer(comm), m_persistent_error_signals{persistent_error_signals} {}
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

  /** Get temp Grad Tensor. */
  AbsDistMatrixType& get_temp_grad() ;
  /** Get transfered inpit for each branch tag **/
  AbsDistMatrixType& get_branch_tag_input(int tag) ;
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

  /** @brief Set whether to keep or dynamically reallocate error signals.
   *
   *  Passing a value of @c true means to keep the error signals; @c
   *  false means to dynamically reallocate them.
   */
  void set_keep_error_signals(bool) override;

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

  /** Setup layer data.
   *  Called by the 'setup' function. Memory is allocated for
   *  distributed matrices.
   */
  void setup_data(size_t max_mini_batch_size) override;

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

  /** @brief Attempt to take ownership of the previous error signal.
   *
   *  If the underlying matrix has the right datatype and
   *  distribution, the signal is moved explicitly. Otherwise a deep
   *  copy is made so that it has the correct datatype and
   *  distribution.
   *
   *  This is valid if the child layer does not have persistent error
   *  signals.
   *
   *  @param child The layer from which the error signal has come.
   *  @param signal The error signal from the layer.
   */
  void move_or_copy_prev_error_signal_(
    const Layer& child,
    std::unique_ptr<El::BaseDistMatrix> signal) final;

  /** @brief Attempt to view the previous error signal.
   *
   *  If the underlying matrix has the right datatype and
   *  distribution, the signal can be viewed directly. Otherwise a
   *  deep copy is made so that it has the correct datatype and
   *  distribution.
   *
   *  This is only valid if the child layer has persistent error
   *  signals. Otherwise, the viewed data my be invalidated.
   *
   *  @param child The layer from which the error signal has come.
   *  @param signal The error signal from the layer.
   */
  void view_or_copy_prev_error_signal_(
    const Layer& child,
    const El::BaseDistMatrix& signal) final;

  /** @brief Deep copy the error signal.
   *
   *  In some cases, it can be determined that neither viewing nor
   *  moving is a possibility. In these cases, we must do a deep copy.
   *
   *  @param child The layer from which the error signal has come.
   *  @param signal The error signal from the layer.
   */
  void deep_copy_prev_error_signal_(
    const Layer& child,
    const El::BaseDistMatrix& signal) final;

  /** @brief Ensure that gradient matrices exist.
   *
   *  This step is performed immediately prior to the bp_compute()
   *  work.
   */
  void allocate_new_gradients_() final;

  /** @brief Send error signals computed by this layer to their
   *         respective parents.
   *
   *  This step is performed immediately after the bp_compute() work
   *  and prior to clearing the previous error signals. This ordering
   *  is necessary in case this layer's error signals are views into
   *  the previous error signals.
   */
  void propagate_error_signals_to_parents_() final;

  /** @brief Free previous error signals, if possible.
   *
   *  This step is performed at the end of a layer's backprop phase.
   */
  void clear_prev_error_signals_() final;

  /** Backward propagation step.
   *  Given the objective function gradients w.r.t. the output
   *  tensors, compute the gradients w.r.t. the input tensors and
   *  w.r.t. the weights. This is essentially an application of the
   *  chain rule.
   */
  void back_prop_impl_() final;

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

  /** Temp grad tensor for Split Layer
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMatrixType>> m_temp_grad;

  /** For Split layer create a tensor for each branch_tag (opt for not transfering data for each branch)
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMatrixType>> m_subgrid_tensors_split;

  /** @brief Whether to keep persistent error signals or dynamically
   *         allocate/deallocate them.
   *
   *  The default behavior is dynamic allocation.
   */
  bool m_persistent_error_signals = false;

#ifdef LBANN_HAS_DISTCONV
  friend class data_type_distconv_adapter<TensorDataType>;
 public:
  data_type_distconv_adapter<TensorDataType>& get_distconv_adapter() override;
  const data_type_distconv_adapter<TensorDataType>& get_distconv_adapter() const override;

 protected:
  void setup_distconv_adapter() override;
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
