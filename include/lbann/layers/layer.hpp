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

#ifndef LBANN_LAYERS_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_LAYER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/distconv.hpp"
#include <lbann.pb.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <array>

namespace lbann {

// Forward declarations
class model;
class weights;
class lbann_callback_sync_layers;

/** Represents a parallel strategy for a layer. */
struct ParallelStrategy {
  /** Number of process groups the sample dimension is split over. */
  int sample_groups = 0;
  /** Number of groups the sample dimension is split over. */
  int sample_splits = 0;
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
  friend class lbann_callback_sync_layers;
  friend class lbann_callback_sync_selected;

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

  /** Human-readable description. */
  virtual description get_description() const;

  /** Forward propagation step.
   *  Apply a mathematical operation to input tensors to obtain output
   *  tensors.
   */
  virtual void forward_prop();
  /** Backward propagation step.
   *  Given the objective function gradients w.r.t. the output
   *  tensors, compute the gradients w.r.t. the input tensors and
   *  w.r.t. the weights. This is essentially an application of the
   *  chain rule.
   */
  virtual void back_prop();
  /** Update step.
   *  Update the layer's internal members. Note that the optimization
   *  step for the weights happens elsewhere.
   */
  virtual bool update();

  virtual void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step);

  /** Setup layer members.
   *  This calls the 'setup_pointers', 'setup_dims', 'setup_matrices',
   *  'setup_data', and 'setup_gpu' (if needed) functions. It is
   *  assumed that pointers to parent/child layers have already been
   *  initialized.
   */
  virtual void setup();
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
  /** Get a human-readable description of the data_layout */
  std::string get_data_layout_string(data_layout d) const;
  /** Get a human-readable description of the device allocation */
  std::string get_device_allocation_string(El::Device dev) const;
  /** Get a short human-readable description of the device allocation */
  std::string get_device_allocation_string_short(El::Device dev) const;

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

  /** Get references to weights. */
  inline std::vector<weights*>& get_weights() { return m_weights; }
  /** Get references to weights. (const) */
  inline const std::vector<weights*>& get_weights() const { return m_weights; }
  /** Set list of pointers to weights. */
  inline void set_weights(std::vector<weights*> w) { get_weights() = w; }
  /** Replace weights with another Layer's weights*/
  void replace_weights(Layer* other_layer);

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
  AbsDistMat& get_activations(int child_index = 0);
  /** Get error signal tensor. */
  AbsDistMat& get_error_signals(int parent_index = 0);
  /** Get previous activation tensor. */
  const AbsDistMat& get_prev_activations(int parent_index = 0) const;
  /** Get activation tensor. */
  const AbsDistMat& get_activations(int child_index = 0) const;
  /** Get previous error signal tensor. */
  const AbsDistMat& get_prev_error_signals(int child_index = 0) const;
  /** Get error signal tensor. */
  const AbsDistMat& get_error_signals(int parent_index = 0) const;
  /** Get local portion of activation tensor. */
  AbsMat& get_local_activations(int child_index = 0);
  /** Get local portion of error signal tensor. */
  AbsMat& get_local_error_signals(int parent_index = 0);
  /** Get local portion of previous activation tensor. */
  const AbsMat& get_local_prev_activations(int parent_index = 0) const;
  /** Get local portion of activation tensor. */
  const AbsMat& get_local_activations(int child_index = 0) const;
  /** Get local portion of previous error signal tensor. */
  const AbsMat& get_local_prev_error_signals(int child_index = 0) const;
  /** Get local portion of error signal tensor. */
  const AbsMat& get_local_error_signals(int parent_index = 0) const;

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

  /** Get the parallel strategy for the layer. */
  ParallelStrategy& get_parallel_strategy() { return m_parallel_strategy; }
  const ParallelStrategy& get_parallel_strategy() const { return m_parallel_strategy; }

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
  virtual std::unique_ptr<AbsDistMat> construct_matrix(const El::Grid& grid,
                                                       std::string type,
                                                       El::Int index);
  /** Setup layer data.
   *  Called by the 'setup' function. Memory is allocated for
   *  distributed matrices.
   */
  virtual void setup_data();
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
  virtual void fp_setup_inputs(El::Int mini_batch_size);
  /** Setup output tensors.
   *  Called by the 'forward_prop' function. Each output tensor is
   *  resized to match the mini-batch size.
   */
  virtual void fp_setup_outputs(El::Int mini_batch_size);
  /** Apply layer operation.
   *  Called by the 'forward_prop' function. Given the input tensors,
   *  the output tensors are populated with computed values.
   */
  virtual void fp_compute() = 0;

  // ===========================================================
  // Back prop step helper functions
  // ===========================================================

  /** Setup gradient w.r.t. output tensors.
   *  Called by the 'back_prop' function. Each gradient w.r.t. output
   *  tensor is setup as a view or copy of the corresponding child
   *  layer's gradient w.r.t. input tensor.
   */
  virtual void bp_setup_gradient_wrt_outputs(El::Int mini_batch_size);
  /** Setup gradient w.r.t. input tensors.
   *  Called by the 'back_prop' function. Each gradient w.r.t. input
   *  tensor is resized to match the mini-batch size.
   */
  virtual void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size);
  /** Compute objective funciton gradients.
   *  Called by the 'back_prop' function. Given the input, output, and
   *  gradient w.r.t. output tensors, the gradient w.r.t. input
   *  tensors are populated with the computed values and the gradients
   *  w.r.t. the weights are sent to the appropriate optimizers.
   */
  virtual void bp_compute();

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

  /** References to layer weights. */
  std::vector<weights*> m_weights;

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


  /** Parallel strategy for the layer. */
  ParallelStrategy m_parallel_strategy;

#ifdef LBANN_HAS_DISTCONV
 public:
  virtual bool using_distconv() const { return false; }
  void enable_distconv();
  void setup_distconv();
  void setup_inter_layer_adaptation();
  void setup_early_termination();
  void setup_keep_original_tensors();
  virtual void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<lbann::dc::Dist, 4>> &dists, 
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed);
  virtual void setup_tensor_distribution_add_adjacent_invariants(
      std::map<const Layer*, std::array<dc::Dist, 4>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants);
  virtual void setup_tensor_distribution_block();
  // TODO: use dists
  virtual void setup_tensors_fwd(const std::array<dc::Dist, 4> &dists) {}
  virtual void setup_prev_activations_tensor(const std::array<dc::Dist, 4> &dists);
  virtual dc::Array4 get_activations_tensor_local_shape() const;
  virtual void setup_activations_tensor(const std::array<dc::Dist, 4> &dists,
                                        bool allocate=true);
  virtual void setup_activations_copyout_tensor(const std::array<dc::Dist, 4> &dists);  
  virtual void setup_tensors_bwd(const std::array<dc::Dist, 4> &dists);
  virtual void setup_prev_error_signals_tensor(const std::array<dc::Dist, 4> &dists);
  virtual void setup_error_signals_tensor(const std::array<dc::Dist, 4> &dists);
  virtual void setup_error_signals_copyout_tensor(const std::array<dc::Dist, 4> &dists);
  virtual dc::Array4 get_prev_activations_overlap() const;
  virtual dc::Array4 get_activations_overlap() const;  
  virtual dc::Array4 get_prev_error_signals_overlap() const;
  virtual dc::Array4 get_error_signals_overlap() const;
  virtual dc::Array4 get_input_decomposition_block() const;
  virtual dc::Array4 get_output_decomposition_block() const;
#if 0
  virtual const dc::Dist &get_prev_activations_distribution() const {
    return m_prev_activations_dist;
  }
  virtual const dc::Dist &get_activations_distribution() const {
    return m_activations_dist;
  }
  virtual const dc::Dist &get_prev_error_signals_distribution() const {
    return m_prev_error_signals_dist;
  }
  virtual const dc::Dist &get_error_signals_distribution() const {
    return m_error_signals_dist;
  }
#endif
  
  // REFACTORING: returning non-const tensor should be protected
  virtual const dc::TensorDev &get_activations_t() const;
  virtual const dc::TensorDev &get_error_signals_t() const;  
  //virtual ConstTensorDev get_activations_const_view() const;
  //virtual ConstTensorDev get_prev_activations_const_view() const;

  bool distconv_enabled() const {
    return m_distconv_enabled;
  }
  void disable_distconv() {
    m_distconv_enabled = false;
  }
  
 protected:
  virtual bool keep_original_input() const { return m_keep_original_input; }
  virtual bool keep_original_output() const { return m_keep_original_output; }
  virtual void fp_setup_distconv(int mini_batch_size);
  virtual void bp_setup_distconv(int mini_batch_size);  
  
  virtual dc::Array4 get_strides() const;

  // Copis and converts input or output tensors when necessary
  void ensure_prev_activations();
  void copy_out_activations();
  void ensure_prev_error_signals();
  void copy_out_error_signals();

  // negative value disables early termination. DISTCONV_EARLY_TERMINATE
  // environment value will override if set.
  int m_exit_count = -1;
  void early_terminate();
  bool early_terminate_last_iteration() const;

  template <typename Tensor>
  void dump_tensor(const Tensor &t, const std::string &path) const {
    if (getenv("DISTCONV_DUMP")) {
      if (early_terminate_last_iteration()) {
        dc::MPIPrintStreamDebug() << "Dumping tensor to "
                                  << path << "\n";
        cudaDeviceSynchronize();
        distconv::dump_tensor(t, path, true);
      }
    }
  }
  void dump_activations() const {
    dump_tensor(m_activations_t, get_name() + "_activations");
  }
  void dump_reference_activations() {
    assert0(dc::tensor::View(
        m_activations_copyout, get_activations().LockedBuffer()));
    dump_tensor(m_activations_copyout,
                get_name() + "_activations_original");
  }
  void dump_error_signals() const {
    dump_tensor(m_error_signals_t, get_name() + "_error_signals");
  }
  void dump_reference_error_signals() {
    assert0(dc::tensor::View(
        m_error_signals_copyout, get_error_signals().LockedBuffer()));
    dump_tensor(m_error_signals_copyout,
                get_name() + "_error_signals_original");
  }
  
  bool m_parent_copy_in_required = false;
  bool m_parent_shuffle_required = false;
  bool m_child_copy_out_required = false;
  bool m_child_shuffle_required = false;  
  dc::Array4 m_input_decomposition_block;
  dc::Array4 m_output_decomposition_block;  
  /** Previous activation tensor */
  dc::TensorDev m_prev_activations_t;
  /** View to Elemental matrix of previous activations */
  dc::TensorDev m_prev_activations_const_view;
  /** Activation tensor */
  dc::TensorDev m_activations_t;
  /** Elemental-format activation matrix */
  dc::TensorDev m_activations_copyout;
  dc::TensorShuffler *m_prev_activations_shuffler = nullptr;
  dc::TensorShuffler *m_prev_activations_shuffler_last_mb[3];
  dc::TensorShuffler *m_activations_shuffler = nullptr;
  dc::TensorShuffler *m_activations_shuffler_last_mb[3];
  /** Previous error signal tensor */
  dc::TensorDev m_prev_error_signals_t;
  /** View to Elemental matrix */
  dc::TensorDev m_prev_error_signals_const_view;
  /** Error signal tensor */
  dc::TensorDev m_error_signals_t;
  /** Elemental-format matrix */
  dc::TensorDev m_error_signals_copyout;
  dc::TensorShuffler *m_prev_error_signals_shuffler = nullptr;
  dc::TensorShuffler *m_prev_error_signals_shuffler_last_mb[3];
  dc::TensorShuffler *m_error_signals_shuffler = nullptr;
  dc::TensorShuffler *m_error_signals_shuffler_last_mb[3];
 private:
  bool m_distconv_enabled = false;
  bool m_keep_original_input = true;
  bool m_keep_original_output = true;
#endif // LBANN_HAS_DISTCONV

private:

  // ===========================================================
  // Private access functions
  // ===========================================================

  /** Get activation tensor corresponding to child layer. */
  const AbsDistMat& get_activations(const Layer& child) const;
  /** Get error signal tensor corresponding to parent layer. */
  const AbsDistMat& get_error_signals(const Layer& parent) const;

  // ===========================================================
  // Private class members
  // ===========================================================

  /** Dimensions of output tensors. */
  std::vector<std::vector<int>> m_output_dims_list;

  /** Input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMat>> m_inputs;
  /** Output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMat>> m_outputs;
  /** Objective function gradients w.r.t. the output tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMat>> m_gradient_wrt_outputs;
  /** Objective function gradients w.r.t. the input tensors.
   *  Each matrix column corresponds to a flattened mini-batch sample.
   */
  std::vector<std::unique_ptr<AbsDistMat>> m_gradient_wrt_inputs;

  /** Hint layer.
   *  During setup, the output tensor dimensions are set to match the
   *  first output tensor of the hint layer. Derived classes may do
   *  more elaborate setup based on the hint layer.
   */
  const Layer* m_hint_layer = nullptr;

};

} // namespace lbann

#endif // LBANN_LAYERS_LAYER_HPP_INCLUDED
