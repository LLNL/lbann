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
#include "lbann/utils/summary.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/typename.hpp"
#include "lbann/utils/distconv.hpp"
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
class model;
class weights;
namespace callback {
class sync_layers;
} // namespace callback

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
      /** Enable subgraph for the layer. */ 
  int enable_subgraph = 0;  
  /** Branch number in the sub graph. */  
  int sub_branch_tag = 0; 
  /** percentage of parent resources to be allocated to this branch. */ 
  int sub_branch_resource_percentage = 0;
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
        replications == ps.replications &&  
        sub_branch_tag == ps.sub_branch_tag &&  
        sub_branch_resource_percentage == ps.sub_branch_resource_percentage &&  
        enable_subgraph == ps.enable_subgraph;
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
     << "/" << ps.sub_branch_tag  
     << "," << ps.sub_branch_resource_percentage  
     << "/" << ps.enable_subgraph
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
  /** Ranks in grid for the sub-graph */  
  std::unique_ptr<std::set <int, std::greater <int> >> subgrid_ranks; 
  El::Int subgrid_number;
  
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
  virtual void setup(size_t max_mini_batch_size, DataReaderMetaData& dr_metadata,const El::Grid& grid);
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

  /** Get parent layers. */
  inline std::vector<const Layer*>& get_parent_layers() { return m_parent_layers; }
  /** Get parent layers. (const) */
  inline const std::vector<const Layer*>& get_parent_layers() const { return m_parent_layers; }
  /** Get child layers. */
  inline std::vector<const Layer*>& get_child_layers() { return m_child_layers; }
  /** Get child layers. (const) */
  inline const std::vector<const Layer*>& get_child_layers() const { return m_child_layers; }

  inline int find_child_layer_index(const Layer* l) const {
    return std::distance(m_child_layers.begin(),
                         std::find(m_child_layers.begin(),
                                   m_child_layers.end(),
                                   l));
  }

  inline int find_parent_layer_index(const Layer* l) const {
    return std::distance(m_parent_layers.begin(),
                         std::find(m_parent_layers.begin(),
                                   m_parent_layers.end(),
                                   l));
  }

  /** Get number of parent layers. */
  inline int get_num_parents() const { return get_parent_layers().size(); }
  /** Get number of child layers. */
  inline int get_num_children() const { return get_child_layers().size(); }

  /** Get names in a particular list of layers */
  static std::string get_layer_names(const std::vector<const Layer*>& list);
  std::string get_child_names() const { return get_layer_names(m_child_layers); }
  std::string get_parent_names() const { return get_layer_names(m_parent_layers); }

  // ===========================================================
  // Layer pointer manipulation functions
  // ===========================================================

  /** Add a parent layer.
   *  Does nothing if parent is a null pointer, the same layer, or
   *  already a parent.
   */
  void add_parent_layer(const Layer* parent);
  /** Add a child layer.
   *  Does nothing if child is a null pointer, the same layer, or
   *  already a child.
   */
  void add_child_layer(const Layer* child);

  /** Remove all parent layers.
   *  Parent layers are not deallocated.
   */
  void clear_parent_layers() { get_parent_layers().clear(); }
  /** Remove all child layers.
   *  Child layers are not deallocated.
   */
  void clear_child_layers() { get_child_layers().clear(); }

  /** Get list of pointers to other layers. */
  virtual std::vector<Layer*> get_layer_pointers();
  /** Set list of pointers to other layers. */
  virtual void set_layer_pointers(std::vector<Layer*> layers);

  // ===========================================================
  // Weights access functions
  // ===========================================================

  /** Set list of pointers to weights. */
  virtual void set_weights(std::vector<weights*>& w) = 0;
  /** Replace weights with another Layer's weights*/
  virtual void replace_weights(Layer* other_layer) = 0;

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
  std::vector<int> get_input_dims(int input_index = 0) const;
  /** Get input tensor size. */
  int get_input_size(int input_index = 0) const;
  /** Get output tensor dimensions. */
  std::vector<int> get_output_dims(int output_index = 0) const;
  /** Get output tensor size. */
  int get_output_size(int output_index = 0) const;

  /** Set output tensor dimensions. */
  void set_output_dims(std::vector<int> dims, int output_index = 0);


  /** Get reference to LBANN communicator. */
  lbann_comm* get_comm() const { return m_comm; }

  // ===========================================================
  // Hint layer access functions
  // ===========================================================

  /** Set hint layer.
   *  Properties of the hint layer are used during the setup
   *  phase. For instance, the output tensor dimensions are set to
   *  match the hint layer's first output tensor.
   */
  void set_hint_layer(const Layer* l) { m_hint_layer = l; }

  /** Get hint layer. */
  const Layer* get_hint_layer() const { return m_hint_layer; }

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

  /** References to parent layers. */
  std::vector<const Layer*> m_parent_layers;
  /** References to child layers. */
  std::vector<const Layer*> m_child_layers;

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
  // Private access functions
  // ===========================================================
  /** Get references to weights. */
  virtual std::vector<weights*> get_weights() = 0;
  /** Get references to weights. (const) */
  virtual std::vector<weights const*> get_weights() const = 0;

  // ===========================================================
  // Private class members
  // ===========================================================

  /** Dimensions of output tensors. */
  std::vector<std::vector<int>> m_output_dims_list;

  /** Hint layer.
   *  During setup, the output tensor dimensions are set to match the
   *  first output tensor of the hint layer. Derived classes may do
   *  more elaborate setup based on the hint layer.
   */
  const Layer* m_hint_layer = nullptr;

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
  void prepare_distconv();
  virtual void setup_distconv_adapter() = 0;
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

inline std::vector<weights*> extract_weights(Layer& l) {
  return l.get_weights();
}

inline std::vector<const weights*> extract_weights(Layer const& l) {
  return l.get_weights();
}

} // namespace lbann

#endif // LBANN_LAYERS_LAYER_HPP_INCLUDED
