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

#ifndef LBANN_LAYERS_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_LAYER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/typename.hpp"
#include "lbann/weights/weights.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV
#include <string>
#include <vector>

/** @brief A utility macro for easily defining default-constructed sub-class
 *  builders.*/
#define LBANN_DEFINE_LAYER_BUILDER(LAYER_NAME)                          \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  std::unique_ptr<Layer> build_##LAYER_NAME##_layer_from_pbuf( \
    lbann_comm*, lbann_data::Layer const&)

/** @brief A utility macro for easily defining "default" builders.
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_LAYER_DEFAULT_BUILDER(LAYER_NAME) \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  std::unique_ptr<Layer> build_##LAYER_NAME##_layer_from_pbuf(          \
    lbann_comm* comm, lbann_data::Layer const&) {                       \
    using LayerType = LAYER_NAME##_layer<TensorDataType, Layout, Device>; \
    return make_unique<LayerType>(comm);                                \
  }

/** @brief A utility macro for easily adding ETI for layer builders
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_LAYER_BUILDER_ETI(LAYER_NAME, T, Device)                  \
  template std::unique_ptr<Layer>                                       \
  build_##LAYER_NAME##_layer_from_pbuf<T,::lbann::data_layout::DATA_PARALLEL,Device>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_##LAYER_NAME##_layer_from_pbuf<T,::lbann::data_layout::MODEL_PARALLEL,Device>( \
    lbann_comm*, lbann_data::Layer const&)

// Forward-declare protobuf classes
namespace lbann_data {
class Layer;
}

namespace lbann {

// Forward declarations
class Layer;
class model;
namespace callback {
class sync_layers;
} // namespace callback

/** @brief Smart pointer to manage ownership of a layer object
 *
 *  This should be treated @b exactly like a @c std::unique_ptr<Layer>
 *  , i.e. there should be exactly one instance per pointer and the
 *  copy constructor and copy-assignment operators should never be
 *  used. Using this like a @c std::shared_ptr may lead to unexpected
 *  behavior.
 *
 *  The @b only reason this is not a @c std::unique_ptr is because
 *  Cereal cannot natively serialize raw pointers, making it hard to
 *  serialize the layer graph. However, it can accommodate @c
 *  std::weak_ptr . In an ideal world, Cereal would support a
 *  non-owning smart pointer to an object in @c std::unique_ptr
 *  (possibly the experimental @c observer_ptr ), but we can make do
 *  by managing layers with @c std::shared_ptr .
 *
 *  @todo Replace with @c std::unique_ptr<Layer> when C++ and Cereal
 *  support @c std::observer_ptr.
 */
using OwningLayerPtr = std::shared_ptr<Layer>;
/** @brief Smart pointer to reference a layer object
 *
 *  See @c OwningLayerPtr
 *
 *  @todo Replace with @c std::observer_ptr<Layer*> when supported by
 *  C++ and Cereal.
 */
using ViewingLayerPtr = std::weak_ptr<Layer>;

/** Represents a parallel strategy for a layer. */
struct ParallelStrategy {
  /** Number of process groups the sample dimension is split over. */
  int sample_groups = 0;
  /** Number of groups the sample dimension is split over. */
  int sample_splits = 0;
  /** Number of process groups the depth dimension is split over. */
  int depth_groups = 0;
  /** Number of groups the depth dimension is split over. */
  int depth_splits = 0;
  /** Number of process groups the height dimension is split over. */
  int height_groups = 0;
  /** Number of groups the height dimension is split over. */
  int height_splits = 0;
  /** Number of process groups the width dimension is split over. */
  int width_groups = 0;
  /** Number of groups the width dimension is split over. */
  int width_splits = 0;
  /** Number of process groups the channel dimension is split over. */
  int channel_groups = 0;
  /** Number of groups the channel dimension is split over. */
  int channel_splits = 0;
  /** Number of process groups the filter dimension is split over. */
  int filter_groups = 0;
  /** Number of groups the filter dimension is split over. */
  int filter_splits = 0;
  /** Number of times the layer is replicated (for FC layers right now). */
  int replications = 0;
  bool operator==(const ParallelStrategy &ps) const {
    return sample_groups == ps.sample_groups &&
        sample_splits == ps.sample_splits &&
        depth_groups == ps.depth_groups &&
        depth_splits == ps.depth_splits &&
        height_groups == ps.height_groups &&
        height_splits == ps.height_splits &&
        width_groups == ps.width_groups &&
        width_splits == ps.width_splits &&
        channel_groups == ps.channel_groups &&
        channel_splits == ps.channel_splits &&
        filter_groups == ps.filter_groups &&
        filter_splits == ps.filter_splits &&
        replications == ps.replications;
  }
  bool operator!=(const ParallelStrategy &ps) const {
    return !(*this == ps);
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const ParallelStrategy &ps) {
  os << "{" << ps.sample_groups
     << "/" << ps.sample_splits
     << ", " << ps.depth_groups
     << "/" << ps.depth_splits
     << ", " << ps.height_groups
     << "/" << ps.height_splits
     << ", " << ps.width_groups
     << "/" << ps.width_splits
     << ", " << ps.channel_groups
     << "/" << ps.channel_splits
     << ", " << ps.filter_groups
     << "/" << ps.filter_splits
     << ", " << ps.replications
     << "}";
  return os;
}

/**
 * @brief Neural network tensor operation.
 *
 * A layer takes input tensors ("previous activations") and applies a
 * mathematical operation to obtain output tensors
 * ("activations"). This operation often has trainable parameters
 * called "weights." The previous activations are recieved from
 * "parent layers" and the activations are sent to "child layers,"
 * making each layer a node in a directed graph. The layer graph and
 * the weights are managed by a neural network model class. A layer
 * should also be able to take objective function gradients w.r.t. the
 * outputs ("previous error signals") and compute the objective
 * function gradients w.r.t. the inputs ("error signals") and
 * w.r.t. the weights. This allows the model to perform automatic
 * differentiation and to apply first-order optimization methods to
 * the weights.
 */
class Layer {
  friend class callback::sync_layers;

public:

  Layer(lbann_comm *comm);
  Layer(const Layer& other);
  Layer& operator=(const Layer& other);
  virtual ~Layer() = default;

  /** Copy function.
   *  This function dynamically allocates memory for a layer instance
   *  and instantiates a copy. The caller is responsible for
   *  deallocating the instance.
   */
  virtual Layer* copy() const = 0;

  /** Get the layer type's name.
   *  A layer type name should be brief, human-readable description of
   *  the layer's mathematical operation.
   */
  virtual std::string get_type() const = 0;
  /** Get the layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  inline std::string get_name() const { return m_name; }
  /** Set the layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  inline void set_name(const std::string name) { m_name = name; }
  /** Get a string representing the layer datatype
   */
  virtual std::string get_datatype_name() const {
    return TypeName<DataType>();
  };

  /** Human-readable description. */
  virtual description get_description() const;

  /** Get the parallel strategy for the layer. */
  inline ParallelStrategy& get_parallel_strategy() {
    return m_parallel_strategy;
  }
  /** Get the parallel strategy for the layer. */
  const ParallelStrategy& get_parallel_strategy() const {
    return m_parallel_strategy;
  }

  /** Forward propagation step.
   *  Apply a mathematical operation to input tensors to obtain output
   *  tensors.
   */
  virtual void forward_prop() {};
  /** Backward propagation step.
   *  Given the objective function gradients w.r.t. the output
   *  tensors, compute the gradients w.r.t. the input tensors and
   *  w.r.t. the weights. This is essentially an application of the
   *  chain rule.
   */
  void back_prop();

  /** Update step.
   *  Update the layer's internal members. Note that the optimization
   *  step for the weights happens elsewhere.
   */
  virtual bool update();

  virtual void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step) = 0;

  /** Setup layer members.
   *  This calls the 'setup_pointers', 'setup_dims', 'setup_matrices',
   *  'setup_data', and 'setup_gpu' (if needed) functions. It is
   *  assumed that pointers to parent/child layers have already been
   *  initialized.
   */
  virtual void setup(size_t max_mini_batch_size, DataReaderMetaData& dr_metadata);
  /** Check that the setup is reasonable. */
  virtual void check_setup();

  /** Get data layout of the data tensors.
   *  We assume that the data layouts of the previous activations,
   *  activations, previous error signals, and error signals are the
   *  same. Each concrete layer that is templated on its data layout
   *  should override this function to return its template parameter.
   */
  virtual data_layout get_data_layout() const = 0;
  /** Get the device allocation for the data tensors.
   *  We assume that the decice allocation of the previous activations,
   *  activations, previous error signals, and error signals are the
   *  same. Each concrete layer that is templated on its device allocation
   *  should override this function to return its template parameter.
   */
  virtual El::Device get_device_allocation() const = 0;

  /** Reset layer stat counters. */
  virtual void reset_counters();

  /** Whether the layer is using a GPU implementation. */
  inline bool using_gpus() const {
#ifdef LBANN_HAS_GPU
    return get_device_allocation() == El::Device::GPU;
#else
    return false;
#endif // LBANN_HAS_GPU
  }

  /** Get expected number of parent layers.
   *  A negative value indicates no limit.
   */
  inline int get_expected_num_parent_layers() const { return m_expected_num_parent_layers; }
  /** Get expected number of child layers.
   *  A negative value indicates no limit.
   */
  inline int get_expected_num_child_layers() const { return m_expected_num_child_layers; }

  /** Return the model that manages this layer. */
  inline model* get_model() const { return m_model; }
  /** Set the model that manages this layer. */
  inline void set_model(model* m) { m_model = m; }

  virtual El::Matrix<El::Int>* get_sample_indices_per_mb() { return nullptr; };

  virtual bool save_to_checkpoint_shared(persist& p) const;
  virtual bool load_from_checkpoint_shared(persist& p);

  virtual bool save_to_checkpoint_distributed(persist& p) const;
  virtual bool load_from_checkpoint_distributed(persist& p);

  /** Write layer to proto file */
  virtual void write_proto(lbann_data::Layer* proto) const;

  const Layer& get_parent_layer(size_t index=0) const;
  const Layer& get_child_layer(size_t index=0) const;

  std::vector<const Layer*> get_parent_layers() const;
  std::vector<const Layer*> get_child_layers() const;

  size_t find_parent_layer_index(const Layer& l) const;
  size_t find_child_layer_index(const Layer& l) const;

  /** Get number of parent layers. */
  size_t get_num_parents() const { return get_parent_layers().size(); }
  /** Get number of child layers. */
  size_t get_num_children() const { return get_child_layers().size(); }

  std::string get_parent_names() const;
  std::string get_child_names() const;

  // ===========================================================
  // Layer pointer manipulation functions
  // ===========================================================

  /** @brief Add a parent layer
   *
   *  Does nothing if parent is a null pointer, the same layer, or
   *  already a parent.
   */
  void add_parent_layer(ViewingLayerPtr parent);
  /** @brief Add a child layer
   *
   *  Does nothing if child is a null pointer, the same layer, or
   *  already a child.
   */
  void add_child_layer(ViewingLayerPtr child);

  void replace_parent_layer(ViewingLayerPtr l, size_t index);
  void replace_child_layer(ViewingLayerPtr l, size_t index);

  /** @brief Remove pointers to parent layers */
  void clear_parent_layers() { get_parent_layers().clear(); }
  /** @brief Remove pointers to child layers */
  void clear_child_layers() { get_child_layers().clear(); }

  ViewingLayerPtr get_parent_layer_pointer(size_t index) const;
  ViewingLayerPtr get_child_layer_pointer(size_t index) const;

  /** @brief List of pointers to other layers */
  virtual std::vector<ViewingLayerPtr> get_layer_pointers();
  /** @brief Set list of pointers to other layers
   *
   *  Input should match output of @c get_layer_pointers .
   */
  virtual void set_layer_pointers(std::vector<ViewingLayerPtr> layers);

  // ===========================================================
  // Weights access functions
  // ===========================================================

  /** Set list of pointers to weights. */
  void set_weights(std::vector<weights*> const& w) {
    m_weights = w;
  }

  /** Replace weights with another Layer's weights*/
  void replace_weights(Layer const& other_layer);

  // ===========================================================
  // Tensor access functions
  // ===========================================================

  /** Get activation tensor corresponding to child layer. */
  virtual const BaseDistMat& get_activations(const Layer& child) const = 0;
  /** Get error signal tensor corresponding to parent layer. */
  virtual const BaseDistMat& get_error_signals(const Layer& parent) const = 0;

  // ===========================================================
  // Tensor dimension access functions
  // ===========================================================

  /** Get input tensor dimensions. */
  std::vector<int> get_input_dims(size_t input_index = 0) const;
  /** Get input tensor size. */
  int get_input_size(size_t input_index = 0) const;
  /** Get output tensor dimensions. */
  std::vector<int> get_output_dims(size_t output_index = 0) const;
  /** Get output tensor size. */
  int get_output_size(size_t output_index = 0) const;

  /** Set output tensor dimensions. */
  void set_output_dims(std::vector<int> dims, size_t output_index = 0);


  /** Get reference to LBANN communicator. */
  lbann_comm* get_comm() const { return m_comm; }

  // ===========================================================
  // Hint layer access functions
  // ===========================================================

  /** Set hint layer.
   *
   *  Properties of the hint layer are used during the setup
   *  phase. For instance, the output tensor dimensions are set to
   *  match the hint layer's first output tensor.
   */
  void set_hint_layer(ViewingLayerPtr l);

  /** Get hint layer. */
  const Layer* get_hint_layer() const;

  // ===========================================================
  // Freeze management functions
  // ===========================================================

  void freeze();
  void unfreeze();
  bool is_frozen() const;

  /** @brief Set whether to keep or dynamically reallocate error signals.
   *
   *  Passing a value of @c true means to keep the error signals; @c
   *  false means to dynamically reallocate them.
   */
  virtual void set_keep_error_signals(bool) = 0;

protected:

  /** @name Weights-related accessors */
  ///@{
  void add_weights(weights* w) {
    m_weights.push_back(w);
  }
  size_t num_weights() const noexcept { return m_weights.size(); }
  bool has_weights() const noexcept { return num_weights() > 0; }
  bool has_weights(size_t idx) const noexcept {
    return ((idx < this->num_weights()) && (m_weights[idx]));
  }
  void set_num_weights(size_t n) { m_weights.resize(n, nullptr); }
  void set_weights(size_t idx, weights* w) {
    m_weights.at(idx) = w;
  }
  weights const& get_weights(size_t idx) const {
    if (idx >= num_weights()) {
      LBANN_ERROR("Asked for weights index \"", idx, "\"; "
                  "however, this layer has ", num_weights(),
                  " weights associated with it.");
    }
    if (m_weights[idx] == nullptr) {
      LBANN_ERROR("Logic error: Detected an in-bounds null weights pointer.");
    }
    return *(m_weights[idx]);
  }

  weights& get_weights(size_t idx) {
    return const_cast<weights&>(
      static_cast<Layer const&>(*this).get_weights(idx));
  }

  void add_as_gradient_source()
  {
    for (auto&& w : this->m_weights) {
      optimizer* opt = w->get_optimizer();
      if (opt != nullptr) { opt->add_gradient_source(this); }
    }
  }

  void remove_as_gradient_source()
  {
    for (auto&& w : this->m_weights) {
      auto&& opt = w->get_optimizer();
      if (opt != nullptr) { opt->remove_gradient_source(this); }
    }
  }
  ///@}

  // ===========================================================
  // Setup helper functions
  // ===========================================================

  /** Setup layer pointers.
   *  Called by the 'setup' function. Pointers to parent/child layers
   *  are assumed to be already initialized.
   */
  virtual void setup_pointers();
  /** Setup tensor dimensions
   *  Called by the 'setup' function. If there are any input tensors,
   *  the base method sets all uninitialized output tensor dimensions
   *  equal to the first input tensor dimensions.
   */
  virtual void setup_dims(DataReaderMetaData& dr_metadata);
  /** Setup distributed matrices.
   *  Called by the 'setup' function. Each column of these distributed
   *  matrices is interpreted as the flattened tensor for a mini-batch
   *  sample. The matrices themselves are constructed by calling the
   *  'construct_matrix' function. If any matrices have already been
   *  setup, they are destroyed and reinstantiated.
   */
  virtual void setup_matrices(const El::Grid& grid) = 0;
  /** Setup layer data.
   *  Called by the 'setup' function. Memory is allocated for
   *  distributed matrices.
   */
  virtual void setup_data(size_t max_mini_batch_size) {};
  /** Setup GPU objects.
   *  Called by the 'setup' function if the layer is on GPUs.
   */
  virtual void setup_gpu() {}

  // ===========================================================
  // Forward prop step helper functions
  // ===========================================================

  /** Setup input tensors.
   *  Called by the 'forward_prop' function. Each input tensor is
   *  setup as a view or copy of the corresponding parent layer's
   *  output tensor.
   */
  virtual void fp_setup_inputs(El::Int mini_batch_size) = 0;
  /** Setup output tensors.
   *  Called by the 'forward_prop' function. Each output tensor is
   *  resized to match the mini-batch size.
   */
  virtual void fp_setup_outputs(El::Int mini_batch_size) = 0;
  /** Apply layer operation.
   *  Called by the 'forward_prop' function. Given the input tensors,
   *  the output tensors are populated with computed values.
   */
  virtual void fp_compute() = 0;

  // ===========================================================
  // Back prop step helper functions
  // ===========================================================

  /** Setup gradient w.r.t. input tensors.
   *  Called by the 'back_prop' function. Each gradient w.r.t. input
   *  tensor is resized to match the mini-batch size.
   */
  virtual void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) = 0;
  /** Compute objective funciton gradients.
   *  Called by the 'back_prop' function. Given the input, output, and
   *  gradient w.r.t. output tensors, the gradient w.r.t. input
   *  tensors are populated with the computed values and the gradients
   *  w.r.t. the weights are sent to the appropriate optimizers.
   */
  virtual void bp_compute() {};

  // ===========================================================
  // Update step helper functions
  // ===========================================================

  /** Perform the computation for the update step.
   *  Returns false if the layer must reset for a new training epoch.
   */
  virtual bool update_compute() { return true; }

  // ===========================================================
  // Protected class members
  // ===========================================================

  /** Reference to LBANN communicator. */
  lbann_comm *m_comm;

  /** Expected number of parent layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_parent_layers = 1;
  /** Expected number of child layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_child_layers = 1;

  /** Reference to model managing this layer. */
  model *m_model = nullptr;

  /** Avoid back prop if frozen */
  bool m_frozen;

  /** Time spent in forward propagation. */
  EvalType m_fp_time;
  /** Time spent in the forward propagation computation. */
  EvalType m_fp_compute_time;
  /** Time spent in backward propagation. */
  EvalType m_bp_time;
  /** Time spent in the backward propagation computation. */
  EvalType m_bp_compute_time;
  /** Time spent in updates. */
  EvalType m_update_time;

  /** Layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  std::string m_name;

private:

  virtual void setup_weights(size_t idx, weights& w) = 0;

  /** @name Implementation details of back-prop. */
  ///@{

  /** @brief Move error signals from a child to its parent.
   *
   *  This is a hacky workaround to C++ rules for protected member
   *  functions. No error-checking is done, e.g., to assert that the
   *  two layers actually have a parent-child relationship because
   *  this is just an implementation detail. The symbol is never
   *  exposed to the public API.
   *
   *  @param parent The parent layer, into which the signal is moved
   *  @param child  The child layer, from which the signal is moved
   *  @param signal The now-released error signal from the child layer
   */
  friend void attempt_move_error_signal(
    Layer& parent, Layer const& child,
    std::unique_ptr<BaseDistMat> signals);
  friend void attempt_view_error_signal(
    Layer& parent, Layer const& child,
    const BaseDistMat& signals);
  friend void deep_copy_error_signal(
    Layer& parent, Layer const& child,
    const BaseDistMat& signals);

  /** @brief Computes the core back-prop steps. */
  virtual void back_prop_impl_() = 0;

  /** @brief Allocates new storage for the gradients that this layer
   *         will compute.
   *
   *  If the layer has persistent error signal information, this will
   *  simply clear the gradients.
   */
  virtual void allocate_new_gradients_() = 0;

  /** @brief Moves all error signals to their respective parents.
   *
   *  Error signals from this instances either are directly moved into
   *  the parent layer or, in cases in which a direct move is not
   *  possible, are deep-copied into a new tensor in the parent layer
   *  (e.g., into a different data type or data distribution).
   */
  virtual void propagate_error_signals_to_parents_() = 0;

  /** @brief Releases the error signals propagated from the child
   *         layers.
   *
   *  At the conclusion of back-prop, the error signals propagated
   *  from the child layers are no longer needed. This ensures that
   *  the memory is released.
   *
   *  This function may do other work, but must respect the persistent
   *  error signal flag.
   */
  virtual void clear_prev_error_signals_() = 0;

  /** @brief Assumes ownership of the error signals from the specified
   *         child layer.
   *
   *  This is a simple pointer move when possible; otherwise it is a
   *  deep-copy of the signal data.
   *
   *  @param child The layer whence the signal is coming.
   *  @param signal The error signals being sent to this layer.
   */
  virtual void move_or_copy_prev_error_signal_(
    const Layer& child,
    std::unique_ptr<El::BaseDistMatrix> signal) = 0;

  /** @brief Attempts to view the error signals from the specified
   *         child layer.
   *
   *  This is a simple data view when possible; otherwise it is a
   *  deep-copy of the signal data.
   *
   *  @param child The layer whence the signal is coming.
   *  @param signal The error signals being sent to this layer.
   */
  virtual void view_or_copy_prev_error_signal_(
    const Layer& child,
    const El::BaseDistMatrix& signal) = 0;

  /** @brief Deep-copy the error signals from the specified child
   *         layer.
   *
   *  @param child The layer whence the signal is coming.
   *  @param signal The error signals being sent to this layer.
   */
  virtual void deep_copy_prev_error_signal_(
    const Layer& child,
    const El::BaseDistMatrix& signal) = 0;

  ///@}

  // ===========================================================
  // Private class members
  // ===========================================================

  /** @brief References to parent layers */
  std::vector<ViewingLayerPtr> m_parent_layers;
  /** @brief References to child layers */
  std::vector<ViewingLayerPtr> m_child_layers;

  /** @brief References to layer weights.
   *
   *  These are references to the base weights objects. The tensor
   *  data type for weights storage might differ from the tensor data
   *  type of this layer's tensors. To ensure consistency, we must
   *  only access weights values through the WeightsProxy class during
   *  training.
   */
  std::vector<weights*> m_weights;

  /** Dimensions of output tensors. */
  std::vector<std::vector<int>> m_output_dims_list;

  /** Hint layer.
   *  During setup, the output tensor dimensions are set to match the
   *  first output tensor of the hint layer. Derived classes may do
   *  more elaborate setup based on the hint layer.
   */
  ViewingLayerPtr m_hint_layer;

  /** Parallel strategy for the layer. */
  ParallelStrategy m_parallel_strategy;

private:
  friend std::vector<const weights*> extract_weights(Layer const& l);
  friend std::vector<weights*> extract_weights(Layer& l);

#ifdef LBANN_HAS_DISTCONV
  friend class distconv_adapter;
 public:
  /** Indicate whether distconv is enabled. */
  bool distconv_enabled() const;
  /** Indicate whether original input matrices need to be set up. */
  virtual bool keep_original_inputs(int index) const;
  /** Indicate whether original output matrices need to be set up. */
  virtual bool keep_original_outputs(int index) const;
  /** Indicate whether original gradient wrt input matrices need to be set up. */
  virtual bool keep_original_gradient_wrt_inputs(int index) const;
  /** Indicate whether original gradient wrt output matrices need to be set up. */
  virtual bool keep_original_gradient_wrt_outputs(int index) const;
  /** Retrievs distconv adapter. */
  virtual const distconv_adapter& get_distconv_adapter() const;
  /** Retrievs distconv adapter. */
  virtual distconv_adapter& get_distconv_adapter();

 protected:
  /** Indicate whether distconv is supported. */
  virtual bool is_distconv_supported() const { return false; }
  /** Pre-initialize distconv attributes needed for setup_data(). */
  void prepare_distconv(const DataReaderMetaData& dr_metadata);
  virtual void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) = 0;
  std::unique_ptr<distconv_adapter>& get_distconv_adapter_ptr() {
    return m_dc; };
  const std::unique_ptr<distconv_adapter>& get_distconv_adapter_ptr() const {
    return m_dc; };

 private:
  mutable bool m_distconv_enabled = false;
  mutable bool m_distconv_enabled_set = false;
  std::unique_ptr<distconv_adapter> m_dc;
#endif // LBANN_HAS_DISTCONV
};

// FIXME (trb 05/28/2020): These should go away. They're used in
// "model.cpp" and "model_factory.cpp" but could be refactored
// out. Outside the scope of current PR.
inline std::vector<weights*> extract_weights(Layer& l) {
  return l.m_weights;
}

inline std::vector<const weights*> extract_weights(Layer const& l) {
  return {l.m_weights.cbegin(), l.m_weights.cend()};
}

} // namespace lbann

#endif // LBANN_LAYERS_LAYER_HPP_INCLUDED
