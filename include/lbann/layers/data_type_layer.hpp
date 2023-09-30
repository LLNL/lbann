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

#ifndef LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/weights/weights_proxy.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#include <array>
#include <map>
#include <set>
#endif // LBANN_HAS_DISTCONV

namespace cereal {
class access;
} // namespace cereal

namespace lbann {

// Forward declarations
namespace dnn_lib {
template <typename U>
class data_parallel_layer_tensor_manager;
template <typename U>
class entrywise_layer_tensor_manager;
} // namespace dnn_lib

using supported_layer_data_type = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  fp16,
#endif
#ifdef LBANN_HAS_HALF
  cpu_fp16,
#endif
  float,
  double>;

template <typename InputTensorDataType,
          typename OutputTensorDataType = InputTensorDataType>
class data_type_layer : public Layer
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using InputAbsDistMatrixType = El::AbstractDistMatrix<InputTensorDataType>;
  using OutputAbsDistMatrixType = El::AbstractDistMatrix<OutputTensorDataType>;

  /** @brief The proxy tensor type expected in this object. */
  template <El::Device D>
  using InputAbsDistMatReadProxyType =
    El::AbstractDistMatrixReadDeviceProxy<InputTensorDataType, D>;
  template <El::Device D>
  using OutputAbsDistMatReadProxyType =
    El::AbstractDistMatrixReadDeviceProxy<OutputTensorDataType, D>;

  /** @brief The local tensor type expected in this object. */
  using InputAbsMatrixType = El::AbstractMatrix<InputTensorDataType>;
  using OutputAbsMatrixType = El::AbstractMatrix<OutputTensorDataType>;

  /** @brief The proxy type for weights used by this object. */
  using WeightsProxyType = weights_proxy<InputTensorDataType>;

  ///@}

public:
  static_assert(
    h2::meta::tlist::MemberV<InputTensorDataType, supported_layer_data_type>(),
    "Must use a supported type.");

  data_type_layer(lbann_comm* /*comm*/, bool persistent_error_signals = false)
    : Layer(), m_persistent_error_signals{persistent_error_signals}
  {}
  virtual ~data_type_layer() = default;

  /** Get a string representing the layer datatype
   */
  std::string get_datatype_name() const override
  {
    return TypeName<OutputTensorDataType>();
  };

  /** Forward propagation step.
   *  Apply a mathematical operation to input tensors to obtain output
   *  tensors.
   */
  void forward_prop() final;

  void summarize_matrices(lbann_summary& summarizer, int step) override;

  /** Check that the setup is reasonable. */
  void check_setup() override;

  // ===========================================================
  // Public Tensor access functions
  // ===========================================================

  /** Get activation tensor corresponding to child layer. */
  const OutputAbsDistMatrixType&
  get_activations(const Layer& child) const override;
  /** Get error signal tensor corresponding to parent layer. */
  const InputAbsDistMatrixType&
  get_error_signals(const Layer& parent) const override;

  /** Get temp Grad Tensor. */
  OutputAbsDistMatrixType& get_temp_grad();
  /** Get transfered input for each branch tag **/
  InputAbsDistMatrixType& get_branch_tag_input(int tag);

  std::vector<std::unique_ptr<InputAbsDistMatrixType>>&
  get_branch_tag_input_vector();

  /** return all activations/errors for a layer
      Used in subgraph parallelism to implement collective communication in
     split, sum, ... layers*/
  std::vector<std::unique_ptr<OutputAbsDistMatrixType>>& get_all_activations();
  std::vector<std::unique_ptr<InputAbsDistMatrixType>>&
  get_all_prev_activations();
  std::vector<std::unique_ptr<OutputAbsDistMatrixType>>&
  get_all_prev_error_signals();
  std::vector<std::unique_ptr<InputAbsDistMatrixType>>& get_all_error_signals();

  /** Get activation tensor. */
  OutputAbsDistMatrixType& get_activations(int child_index = 0);
  /** Get error signal tensor. */
  InputAbsDistMatrixType& get_error_signals(int parent_index = 0);
  /** Get activation tensor. */
  const OutputAbsDistMatrixType& get_activations(int child_index = 0) const;
  /** Get error signal tensor. */
  const InputAbsDistMatrixType& get_error_signals(int parent_index = 0) const;

  /** Get local portion of activation tensor. */
  OutputAbsMatrixType& get_local_activations(int child_index = 0);
  /** Get local portion of error signal tensor. */
  InputAbsMatrixType& get_local_error_signals(int parent_index = 0);
  /** Get local portion of activation tensor. */
  const OutputAbsMatrixType& get_local_activations(int child_index = 0) const;
  /** Get local portion of error signal tensor. */
  const InputAbsMatrixType& get_local_error_signals(int parent_index = 0) const;

  /** @brief Set whether to keep or dynamically reallocate error signals.
   *
   *  Passing a value of @c true means to keep the error signals; @c
   *  false means to dynamically reallocate them.
   */
  void set_keep_error_signals(bool) override;

  El::mpi::Comm& get_subgrid_comm() { return *m_interSubGridVCComm; }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** @brief Protected lifecycle functions */
  ///@{
  data_type_layer(data_type_layer&& other) = default;
  data_type_layer(data_type_layer const& other);

  data_type_layer& operator=(data_type_layer&& other) = default;
  data_type_layer& operator=(data_type_layer const& other);
  ///@}

  // ===========================================================
  // Protected Tensor access functions
  // ===========================================================

  /** Get previous activation tensor. */
  InputAbsDistMatrixType& get_prev_activations(int parent_index = 0);
  /** Get previous error signal tensor. */
  OutputAbsDistMatrixType& get_prev_error_signals(int child_index = 0);

  /** Get previous activation tensor. */
  const InputAbsDistMatrixType&
  get_prev_activations(int parent_index = 0) const;
  /** Get previous error signal tensor. */
  const OutputAbsDistMatrixType&
  get_prev_error_signals(int child_index = 0) const;

  /** Get local portion of previous activation tensor. */
  const InputAbsMatrixType&
  get_local_prev_activations(int parent_index = 0) const;
  /** Get local portion of previous error signal tensor. */
  const OutputAbsMatrixType&
  get_local_prev_error_signals(int child_index = 0) const;

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
  void setup_matrices(const std::vector<El::Grid*>& grids) override;

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

  /** @brief Get the values matrix for a specific weights object */
  InputAbsDistMatrixType const& weights_values(size_t idx) const;

  /** @brief Get a specific master weights object.
   *
   *  This is sufficient for setting or accessing metadata about the
   *  weights class.
   */
  weights& master_weights(size_t idx) { return get_weights(idx); }
  weights const& master_weights(size_t idx) const { return get_weights(idx); }

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
  void view_or_copy_prev_error_signal_(const Layer& child,
                                       const El::BaseDistMatrix& signal) final;

  /** @brief Deep copy the error signal.
   *
   *  In some cases, it can be determined that neither viewing nor
   *  moving is a possibility. In these cases, we must do a deep copy.
   *
   *  @param child The layer from which the error signal has come.
   *  @param signal The error signal from the layer.
   */
  void deep_copy_prev_error_signal_(const Layer& child,
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
  // Private class members
  // ===========================================================

  /** @brief Persistent, read-only, proxied views of the weights
   *         values matrix.
   *
   *  @note (trb 05/28/2020): These are kept as members out of
   *  consideration for the case where accessing them could require a
   *  deep copy. This is more out of my own concern about ways in
   *  which derived classes could abuse weights; in theory, I believe,
   *  you could just create these on the fly once during FP and once
   *  during BP. Then the question is: does the performance cost of
   *  (potentially) two(ish) copies or the memory cost of storing an
   *  additional copy of the weights hurt more?
   */
  std::vector<WeightsProxyType> m_weights_proxy;

  /** Input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<InputAbsDistMatrixType>> m_inputs;
  /** Output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<OutputAbsDistMatrixType>> m_outputs;
  /** Objective function gradients w.r.t. the output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<OutputAbsDistMatrixType>> m_gradient_wrt_outputs;
  /** Objective function gradients w.r.t. the input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<InputAbsDistMatrixType>> m_gradient_wrt_inputs;

  /** Temp grad tensor for Split Layer
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<OutputAbsDistMatrixType>> m_temp_grad;

  /** For Split layer create a tensor for each branch_tag (opt for not
   * transfering data for each branch) Each matrix column corresponds to a
   * flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<OutputAbsDistMatrixType>> m_subgrid_tensors_split;

  /** @brief Whether to keep persistent error signals or dynamically
   *         allocate/deallocate them.
   *
   *  The default behavior is dynamic allocation.
   */
  bool m_persistent_error_signals = false;

#ifdef LBANN_HAS_DISTCONV
  friend class data_type_distconv_adapter<InputTensorDataType,
                                          OutputTensorDataType>;

public:
  data_type_distconv_adapter<InputTensorDataType, OutputTensorDataType>&
  get_distconv_adapter() override;
  const data_type_distconv_adapter<InputTensorDataType, OutputTensorDataType>&
  get_distconv_adapter() const override;

protected:
  void setup_distconv_adapter() override;
#endif // LBANN_HAS_DISTCONV

#ifdef LBANN_HAS_GPU
  template <typename U>
  friend class dnn_lib::data_parallel_layer_tensor_manager;
  template <typename U>
  friend class dnn_lib::entrywise_layer_tensor_manager;
#endif // LBANN_HAS_GPU
};

#ifndef LBANN_DATA_TYPE_LAYER_INSTANTIATE
#define PROTO(T) extern template class data_type_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#endif // LBANN_DATA_TYPE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_LAYER_HPP_INCLUDED
