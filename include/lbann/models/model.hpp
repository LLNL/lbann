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

#ifndef LBANN_MODELS_MODEL_HPP_INCLUDED
#define LBANN_MODELS_MODEL_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/threads/thread_pool.hpp"

#ifdef LBANN_HAS_ONNX
#include <onnx/onnx_pb.h>
#endif // LBANN_HAS_ONNX

// Note (trb): There's what is, IMO, an STL error in GCC in which the
// dtor for unique_ptr is checking sizeof(T), so this must be a
// complete type. Sigh. (The greater implication of this is that you
// cannot have `unique_ptr<IncompleteType>` as a drop-in for
// `IncompleteType*`, which is annoying.)
#include "lbann/proto/optimizers.pb.h"

#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare protobuf class
namespace lbann_data {
class Model;
}

// Forward declaration
namespace cereal {
class access;
}

namespace lbann {

// Forward declarations
class lbann_comm;
class description;
class Layer;
class lbann_callback;
class Layer;
class TrainingAlgorithm;
class callback_base;
class metric;
class weights;
class optimizer;
class objective_function;
class ExecutionContext;
class persist;
using OwningWeightsPtr = std::shared_ptr<weights>;
using ViewingWeightsPtr = std::weak_ptr<weights>;
using OwningLayerPtr = std::shared_ptr<Layer>;
using ViewingLayerPtr = std::weak_ptr<Layer>;

/** @brief Abstract base class for neural network models. */
class model
{
public:
  // ===========================================
  // Life cycle functions
  // ===========================================

  model(lbann_comm* comm,
        std::unique_ptr<objective_function> obj_fn,
        std::unique_ptr<lbann_data::Optimizer> default_optimizer_msg = nullptr);
  model(const model& other);
  model& operator=(const model& other);
  ~model() = default;

  /** @brief Metadata Accessors */
  ///@{

  /** @brief Model instance name.
   *  @details Each model in a trainer should have a unique, and
   *  preferably human-readable, name.
   */
  void set_name(std::string name);

  /** @brief Model instance name.
   *  @details Each model in a trainer should have a unique, and
   *  preferably human-readable, name.
   */
  std::string get_name() const noexcept;

  /** @brief Human-readable description. */
  description get_description() const;

  /** @brief Get the model's comm. */
  lbann_comm* get_comm() const noexcept;

  ///@}
  /** @brief Machine-learning object accessors */
  ///@{

  /** @brief Size of model's list of layers. */
  El::Int get_num_layers() const noexcept;
  /** @param pos Position in model's list of layers. */
  Layer& get_layer(El::Int pos);
  /** @param pos Position in model's list of layers. */
  Layer const& get_layer(El::Int pos) const;
  /** @brief Return list of layers in model.
   *  @details The list is in execution order for forward propagation.
   */
  std::vector<Layer*> get_layers();
  /** @brief Return list of layers in model.
   *  @details The list is in execution order for forward propagation.
   */
  std::vector<Layer const*> get_layers() const;
  std::vector<weights*> get_weights();
  std::vector<weights const*> get_weights() const;
  std::vector<ViewingWeightsPtr> get_weights_pointers() const;

  /** @brief Mathematical function to be minimized during training. */
  observer_ptr<objective_function const>
  get_objective_function() const noexcept;

  observer_ptr<objective_function> get_objective_function() noexcept;

  /** @brief Return the model's metrics. */
  std::vector<metric*> get_metrics();
  std::vector<metric const*> get_metrics() const;

#ifdef LBANN_HAS_ONNX
  /** @brief Serialize model to Onnx format */
  void serialize_to_onnx(onnx::ModelProto& mp);
#endif // LBANN_HAS_ONNX

  // ===========================================
  // Model specification
  // ===========================================
  ///@}
  /** @name Model specification */
  ///@{

  /** @brief Add layer to model. */
  void add_layer(OwningLayerPtr&& l);

  /** @brief Add weights to model. */
  void add_weights(OwningWeightsPtr&& w);

  /** @brief Remove weights from model. */
  void remove_weights(std::string const& name);

  /** @brief Register a new callback for the model. */
  void add_callback(std::shared_ptr<callback_base> cb);

  /** @brief Register a new metric for the model. */
  void add_metric(std::unique_ptr<metric> m);

  /** @brief Insert layer in model. */
  void insert_layer(OwningLayerPtr&& l, std::string const& parent_name);

  /** @brief Remove layer from model. */
  void remove_layer(std::string const& name);

  /** @brief Replace layer in model. */
  void replace_layer(OwningLayerPtr&& l, std::string const& name);

  void swap_layers(model& other);
  void swap_weights(model& other);
  void swap_metrics(model& other);
  void swap_objective_function(model& other);

  ///@}

  /** @brief Copy trained weights from input parameter w.
   *
   *  Only weight values are placed, pointers and layer structure are in place.
   *  Weights to be copied are of the same name
   */
  void copy_trained_weights_from(std::vector<weights*>& w);

  /** @brief Construct an instance of the default optimizer.
   *
   *  If there is no default optimizer, a null pointer is returned.
   */
  template <typename TensorDataType>
  std::unique_ptr<optimizer> create_optimizer() const;

  /** @brief Set a flag that can be used to enable / disable the
   *         background I/O activities
   */
  void allow_background_io_activity(bool enable) noexcept;

  /** @brief Are background I/O activities enabled by the input layers */
  bool background_io_activity_allowed() const noexcept;

  // ===========================================
  // Setup
  // ===========================================

  /** @details Must be called after model specification and before
   *  execution. */
  void setup(size_t max_mini_batch_size,
             const std::vector<El::Grid*>& grids,
             bool force = false);

  /** @name Summarization */
  ///@{

  /** @brief Summarize statistics (e.g. timers, counters).
   *  @details These should be computable quickly.
   */
  void summarize_stats(lbann_summary& summarizer);

  /** @brief Summarize matrices (e.g. means).
   *  @details These are called less frequently and can be more expensive.
   */
  void summarize_matrices(lbann_summary& summarizer);

  ///@}
  /** @name Checkpointing and serialization. */
  ///@{

  /** @brief Serialization for checkpoint and restart with Cereal. */
  template <class Archive>
  void serialize(Archive& ar);

  /** @brief Checkpoint model to given file descriptor, return number of bytes
   * written */
  bool save_to_checkpoint_shared(persist& p);
  /** @brief Restore model by reading checkpoint from given file descriptor,
   * return number of bytes read */
  bool load_from_checkpoint_shared(persist& p);

  bool save_to_checkpoint_distributed(persist& p);
  bool load_from_checkpoint_distributed(persist& p);

  /** @brief Write model to proto file */
  void write_proto(lbann_data::Model& proto);

  /** @brief Saves the model explicitly if the save_model callback is present.
   *
   * @deprecated This function both holds on to the notion that models
   *             support callbacks (the majority of those in the
   *             current iteration of callbacks should be thought of
   *             as extensions to training algorithms rather than
   *             extensions of models) and is only used by the
   *             "cycgan" and "aecycgan" drivers, which themselves are
   *             not well-supported.
   */
  void save_model();

  ///@}
  /** @brief Subgraph Parallelism Interface */
  ///@{

  void set_subgrid_communication_type(int type) noexcept;
  int get_subgrid_communication_type() const noexcept;
  void set_subgraph_num_parent_resources(int num_resources) noexcept;
  int get_subgraph_num_parent_resources() const noexcept;
  void set_subgrid_topology(bool type) noexcept;
  bool get_subgrid_topology() const noexcept;
  void enable_subgraph_parallelism() noexcept;
  bool is_subgraph_parallelism_enabled() const noexcept;
  int get_num_resources_non_branch_layers() const noexcept;
  int get_num_resources_branch_layers() const noexcept;
  void set_num_resources_non_branch_layers(int num) noexcept;
  void set_num_resources_branch_layers(int num) noexcept;

  ///@}

private:
  /** @brief Setup-related implementation */
  ///@{

  /** @brief Reorder layer list with a gather.
   *
   *  The new layer list is the same length as @c gather_indices and
   *  its entries are given by
   *  @f[ \text{new\_list}[i] = \text{old\_list}[\text{gather\_indices}[i]] @f]
   *
   *  Since entries in the layer list must be unique, this will fail
   *  if @c gather_indices has any repeated entries.
   */
  void reorder_layers(const std::vector<El::Int>& gather_indices);

  /** @brief Remap pointers.
   *
   *  Layer and weights pointers are remapped using the provided
   *  maps. If a pointer is not a key in the corresponding map, the
   *  pointer is not changed.
   */
  void remap_pointers(
    const std::unordered_map<Layer*, ViewingLayerPtr>& layer_map,
    const std::unordered_map<weights*, ViewingWeightsPtr>& weights_map);

  /** @brief Set up topology of layer graph.
   *
   *  Called in setup function. All layers in connected component of
   *  layer graph are added to the model and all parent/child
   *  relationships between layers are reciprocated.
   */
  void setup_layer_topology();

  /** @brief Set up layer execution order.
   *
   *  Called in setup function. A topological sort applied is to the
   *  layer list so that we can traverse the directed acyclic graph
   *  without violating dependencies.
   */
  void setup_layer_execution_order();

  /** @brief Set up grid tags for all layers.
   *
   *  Called in setup function.
   */

  void setup_layer_grid_tags(const std::vector<El::Grid*>& grids);

  /** @brief Set up layers.
   *
   *  Called in setup function.
   */
  void setup_layers(size_t max_mini_batch_size,
                    const std::vector<El::Grid*>& grids);

  /** @brief Set up weights.
   *
   *  Called in setup function. All weights being used by layers or
   *  the objective function are added to the model and all unused
   *  weights are deleted.
   */
  void setup_weights();

  ///@}
  /** @name Subgraph parallelism implementation */
  ///@{

  /** @brief Setup sub grids for the sub graph parallelism */
  void setup_subgrids();

  void get_subgrids_order(std::vector<int>& ranks_order, int num_branches);
  int get_max_subgraph_branches();
  void check_subgraph_parallelism();
  void setup_subgrid_layers_run_condition();
  void get_parent_subgrid_tags(int layer_index);
  void get_subgraph_subgrids_ranks(std::vector<int>& parent_ranks,
                                   std::vector<int>& subgrid_ranks,
                                   int layer_index,
                                   int number_ranks_in_grid);
  void get_resources_for_spliting_point(std::vector<int>& parent_ranks,
                                        std::vector<int>& subgrid_ranks,
                                        int layer_index,
                                        int number_ranks_in_grid,
                                        int num_subgrids);
  void get_resources_for_merge_layers(std::set<int>& pooled_set,
                                      int child_index,
                                      int num_subgrids);
  void get_resources_for_input_layer(std::vector<int>& masterSubGrid,
                                     int num_subgrids);
  void setup_subcommunicators(const std::vector<El::Grid*>& grids);

  ///@}

public:
  // ===========================================
  // Execution
  // ===========================================

  /** @brief Get the list of callbacks for the model. */
  std::vector<observer_ptr<callback_base>> get_callbacks();

  std::vector<std::shared_ptr<callback_base>>&
  get_callbacks_with_ownership() noexcept;

  /** Check to see if there is a valid training context for the model */
  bool has_valid_execution_context() const noexcept;

  /** Grab the training context of the model */
  ExecutionContext const& get_execution_context() const;

  /** Grab the training context of the model */
  ExecutionContext& get_execution_context();

  /** @brief Reset model pointer and execution mode. */
  void reset_mode(ExecutionContext& context, execution_mode mode);
  /** @brief Reset model statistics for an epoch. */
  void reset_epoch_statistics(execution_mode mode);

  /** @brief Forward propagation step. */
  void forward_prop(execution_mode mode);
  /** @brief Backward propagation step. */
  void backward_prop(bool compute_weight_grads_only = true);
  /** Evaluate any metrics in the model */
  void evaluate_metrics(execution_mode mode, size_t current_mini_batch_size);
  /** @brief Clear each optimizer's gradient.
   *
   *  This must be called before training forward prop since layers
   *  set an optimizer flag during forward prop.
   */
  void clear_gradients();
  /** @brief Update weights step. */
  void update_weights();
  /** @brief Update layers step. */
  bool update_layers();
  /** @brief Reconcile weight values.
   *
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_weight_values();

  // ===========================================
  // Callbacks
  // ===========================================

  /** @brief Execute callbacks at end of setup. */
  void do_setup_end_cbs();
  /** @brief Execute callbacks at start of model forward propagation. */
  void do_model_forward_prop_begin_cbs(execution_mode mode);
  /** @brief Execute callbacks at end of model forward propagation. */
  void do_model_forward_prop_end_cbs(execution_mode mode);
  /** @brief Execute callbacks at start of layer forward propagation. */
  void do_layer_forward_prop_begin_cbs(execution_mode mode, Layer* l);
  /** @brief Execute callbacks at end of layer forward propagation. */
  void do_layer_forward_prop_end_cbs(execution_mode mode, Layer* l);
  /** @brief Execute callbacks at start of model backward propagation. */
  void do_model_backward_prop_begin_cbs();
  /** @brief Execute callbacks at end of model backward propagation. */
  void do_model_backward_prop_end_cbs();
  /** @brief Execute callbacks at start of layer backward propagation. */
  void do_layer_backward_prop_begin_cbs(Layer* l);
  /** @brief Execute callbacks at end of layer backward propagation. */
  void do_layer_backward_prop_end_cbs(Layer* l);
  /** @brief Execute callbacks at start of model optimization. */
  void do_model_optimize_begin_cbs();
  /** @brief Execute callbacks at end of model optimization. */
  void do_model_optimize_end_cbs();
  /** @brief Execute callbacks at the start of weight optimization. */
  void do_weight_optimize_begin_cbs(weights* w);
  /** @brief Execute callbacks at the end of weight optimization. */
  void do_weight_optimize_end_cbs(weights* w);
  /** @brief Return the maximum mini-batch size. */
  size_t get_max_mini_batch_size() const noexcept;
  /** @brief Return the current mini-batch size. */
  El::Int get_current_mini_batch_size() const noexcept;
  /** @brief Set the current mini-batch size. */
  void set_current_mini_batch_size(El::Int) noexcept;

private:
  friend cereal::access;
  model();

private:
  // map to store all distinct grids in the model
  std::unordered_map<std::string, std::shared_ptr<El::Grid>> grids;

  std::unordered_map<std::string, std::shared_ptr<El::mpi::Comm>>
    subCommunicatorsSubgrids;
  // map to store all distinct mpi groups in the model (one to one mapping with
  // grids)
  std::unordered_map<std::string, std::unique_ptr<El::mpi::Group>>
    grids_mpi_groups;

private:
  /** Pointer to the execution context object used for training or evaluating
   * this model */
  observer_ptr<ExecutionContext> m_execution_context;

  /** @brief LBANN communicator. */
  lbann_comm* m_comm;

  /*experimental code for Sub graph*/
  /** Enable vector communication for the subgraph parallelism */
  // 0: send-recv based subgrid communication
  // 1: collective based subgrid communication without optimization that
  // requires specific assumptions like subgrids should have same size and
  // creates sub-communicators everytime 2: collective based subgrid
  // communication with optimization

  int vector_communication_subgraph = 0;

  // Number of resources for parent (common) grid
  // 0: use all resources (default)
  int subgraph_num_resources_parent = 0;

  // 0: no topology aware design
  // 1: master grid in round robin manner of nodes (GPUs per node 4)  1 3 5 7, 2
  // 4 6 8
  bool enable_subgraph_topology = false;

  // whether subgraph parallelism is enabled or not for the model
  bool apply_subgraph_parallelism = false;

  // total number of resources / ranks for branch (subgrid) layers
  int num_resources_branch_layers;

  // total number of resources / ranks for common/seq layers
  int num_resources_non_branch_layers;

  /** @brief Model instance's name.
   *  @details Each model in a trainer should have a unique,
   *  preferably human-readable, name.
   */
  std::string m_name;

  /** @brief Tensor operations.
   *  @details The list is in execution order for forward propagation.
   */
  std::vector<OwningLayerPtr> m_layers;

  /** @brief Trainable parameters. */
  std::vector<OwningWeightsPtr> m_weights;

  /** @details If a layer needs to construct an optimizer during
   *  setup, it will make a copy of the default optimizer. This object
   *  is just used to create copies and is not actually used for
   *  optimization.
   */
  std::unique_ptr<lbann_data::Optimizer> m_default_optimizer_msg;

  /** @brief Mathematical function to be minimized during training. */
  std::unique_ptr<objective_function> m_objective_function;

  /** @brief Numerical quantities to evaluate model performance.
   *  @details Does not affect training.
   */
  std::vector<std::unique_ptr<metric>> m_metrics;

  /** @brief Current callbacks to process. */
  std::vector<std::shared_ptr<callback_base>> m_callbacks;

  /** @brief Flag that allows input layers to fetch data in the background */
  bool m_background_io_allowed = true;

  /** @brief Is the model setup
   *  @details Flag to indicate if the setup function has been called
   */
  bool m_model_is_setup = false;

private:
  // ===========================================
  // Functions to add utility layers
  // ===========================================

  /** @brief Insert evaluation layers where needed.
   *
   *  If a @c lbann::layer_term or @c lbann::layer_metric corresponds
   *  to a layer that is not an evaluation_layer, an evaluation layer
   *  is created and added to the model.
   *
   *  @param layer_set      Layers in model. Updated with any newly
   *                        created layers.
   *  @param layer_names    Names of layers in model. Updated with any
   *                        newly created layers.
   */
  void add_evaluation_layers(std::unordered_set<Layer*>& layer_set,
                             std::unordered_set<std::string>& layer_names);

  /** @brief Insert dummy layers after layers with too few children.
   *
   *  If a layer expects more child layers than it has, add dummy
   *  layers until it has enough children.
   *
   *  @param layer_names    Names of layers in model. Updated with any
   *                        newly created layers.
   */
  void add_dummy_layers(std::unordered_set<std::string>& layer_names);
  /** @brief Insert split layers after layers with too many children.
   *
   *  If a layer expects one child layer but has multiple, add a split
   *  layer to the model.
   *
   *  @param layer_names    Names of layers in model. Updated with any
   *                        newly created layers.
   */
  void add_split_layers(std::unordered_set<std::string>& layer_names);

  void ensure_input_layers_first();

  /** @brief The maximum mini-batch size.
   *  @details This should be set before setup_distconv() is called.
   */
  size_t m_max_mini_batch_size;

  /** @brief The current mini-batch size.
   *  @details This should be set on each step by the execution
   *  algorithm using the value that the data coordinator gets from
   *  the data readers.
   *
   *  Number of samples being processed in the current step (iteration),
   *  used for correctly averaging gradients.

   */
  El::Int m_current_mini_batch_size;

#ifdef LBANN_HAS_DISTCONV
private:
  void setup_distconv();
  void setup_distributions();
  void print_distributions() const;
#endif // LBANN_HAS_DISTCONV
};     // class model

inline std::string model::get_name() const noexcept { return m_name; }

inline observer_ptr<objective_function> model::get_objective_function() noexcept
{
  return m_objective_function.get();
}

inline observer_ptr<objective_function const>
model::get_objective_function() const noexcept
{
  return m_objective_function.get();
}

inline std::vector<observer_ptr<callback_base>> model::get_callbacks()
{
  std::vector<observer_ptr<callback_base>> callback_list;
  callback_list.reserve(m_callbacks.size());
  for (const auto& ptr : m_callbacks) {
    callback_list.push_back(ptr.get());
  }
  return callback_list;
}

inline std::vector<std::shared_ptr<callback_base>>&
model::get_callbacks_with_ownership() noexcept
{
  return m_callbacks;
}

inline lbann_comm* model::get_comm() const noexcept { return m_comm; }

inline bool model::has_valid_execution_context() const noexcept
{
  return (m_execution_context != nullptr);
}

inline ExecutionContext const& model::get_execution_context() const
{
  if (m_execution_context == nullptr) {
    LBANN_ERROR("execution context is not set");
  }
  return *m_execution_context;
}

inline ExecutionContext& model::get_execution_context()
{
  return const_cast<ExecutionContext&>(
    static_cast<const model&>(*this).get_execution_context());
}

template <typename TensorDataType>
inline std::unique_ptr<optimizer> model::create_optimizer() const
{
  if (m_default_optimizer_msg)
    return proto::construct_optimizer<TensorDataType>(*m_default_optimizer_msg);
  return nullptr;
}

inline void model::allow_background_io_activity(bool enable) noexcept
{
  m_background_io_allowed = enable;
}

inline bool model::background_io_activity_allowed() const noexcept
{
  return m_background_io_allowed;
}

inline void model::set_subgrid_communication_type(int type) noexcept
{
  vector_communication_subgraph = type;
}

inline int model::get_subgrid_communication_type() const noexcept
{
  return vector_communication_subgraph;
}

inline void model::set_subgraph_num_parent_resources(int num_resources) noexcept
{
  subgraph_num_resources_parent = num_resources;
}

inline int model::get_subgraph_num_parent_resources() const noexcept
{
  return subgraph_num_resources_parent;
}

inline void model::set_subgrid_topology(bool type) noexcept
{
  enable_subgraph_topology = type;
}

inline bool model::get_subgrid_topology() const noexcept
{
  return enable_subgraph_topology;
}

inline void model::enable_subgraph_parallelism() noexcept
{
  apply_subgraph_parallelism = true;
}

inline bool model::is_subgraph_parallelism_enabled() const noexcept
{
  return apply_subgraph_parallelism;
}

inline int model::get_num_resources_non_branch_layers() const noexcept
{
  return num_resources_non_branch_layers;
}

inline int model::get_num_resources_branch_layers() const noexcept
{
  return num_resources_branch_layers;
}

inline void model::set_num_resources_non_branch_layers(int num) noexcept
{
  num_resources_non_branch_layers = num;
}

inline void model::set_num_resources_branch_layers(int num) noexcept
{
  num_resources_branch_layers = num;
}

inline size_t model::get_max_mini_batch_size() const noexcept
{
  return m_max_mini_batch_size;
}

inline El::Int model::get_current_mini_batch_size() const noexcept
{
  return m_current_mini_batch_size;
}

inline void model::set_current_mini_batch_size(El::Int mini_batch_size) noexcept
{
  if (mini_batch_size > static_cast<El::Int>(m_max_mini_batch_size)) {
    LBANN_WARNING(
      "LOGICAL ERROR: the current mini-batch size ",
      mini_batch_size,
      " is being set to larger than the established maximim mini-batch size ",
      m_max_mini_batch_size,
      ".  Note that this should work properly as all matrices will be resized, "
      "but this is a logical error as the maximum mini-batch size should be "
      "established at setup time to avoid dynamic allocation.");
  }
  m_current_mini_batch_size = mini_batch_size;
  return;
}

} // namespace lbann

#endif // LBANN_MODELS_MODEL_HPP_INCLUDED
