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

#ifndef LBANN_LAYERS_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_LAYER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/typename.hpp"
#include <string>
#include <vector>
#ifdef LBANN_HAS_ONNX
#include <onnx/onnx_pb.h>
#endif

/** @brief A utility macro for easily defining default-constructed sub-class
 *  builders.*/
#define LBANN_DEFINE_LAYER_BUILDER(LAYER_NAME)                                 \
  template <typename TensorDataType, data_layout Layout, El::Device Device>    \
  std::unique_ptr<Layer> build_##LAYER_NAME##_layer_from_pbuf(                 \
    lbann_comm*,                                                               \
    lbann_data::Layer const&)

/** @brief A utility macro for easily defining "default" builders.
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_LAYER_DEFAULT_BUILDER(LAYER_NAME)                                \
  template <typename TensorDataType, data_layout Layout, El::Device Device>    \
  std::unique_ptr<Layer> build_##LAYER_NAME##_layer_from_pbuf(                 \
    lbann_comm* comm,                                                          \
    lbann_data::Layer const&)                                                  \
  {                                                                            \
    using LayerType = LAYER_NAME##_layer<TensorDataType, Layout, Device>;      \
    return std::make_unique<LayerType>(comm);                                  \
  }

/** @brief A utility macro for easily adding ETI for layer builders
 *  @note Must be called inside lbann namespace.
 */
#define LBANN_LAYER_BUILDER_ETI(LAYER_NAME, T, Device)                         \
  template std::unique_ptr<Layer>                                              \
    build_##LAYER_NAME##_layer_from_pbuf<T,                                    \
                                         ::lbann::data_layout::DATA_PARALLEL,  \
                                         Device>(lbann_comm*,                  \
                                                 lbann_data::Layer const&);    \
  template std::unique_ptr<Layer>                                              \
    build_##LAYER_NAME##_layer_from_pbuf<T,                                    \
                                         ::lbann::data_layout::MODEL_PARALLEL, \
                                         Device>(lbann_comm*,                  \
                                                 lbann_data::Layer const&)

// Forward-declare protobuf classes
namespace lbann_data {
class Layer;
}

namespace lbann {

// Forward declarations
class lbann_comm;
class description;
class Layer;
class model;
class lbann_summary;
class weights;
using ViewingWeightsPtr = std::weak_ptr<weights>;
#ifdef LBANN_HAS_DISTCONV
class distconv_adapter;
#endif // LBANN_HAS_DISTCONV
namespace callback {
class sync_layers;
class memory_profiler;
} // namespace callback
class KFAC;
template <hydrogen::Device Device>
class kfac_block_fc_conv;
template <hydrogen::Device Device>
class kfac_block_channelwise_fc;
template <hydrogen::Device Device>
class kfac_block_bn;
template <hydrogen::Device Device>
class kfac_block_gru;

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
 *  support @c std::observer_ptr .
 */
using OwningLayerPtr = std::shared_ptr<Layer>;
/** @brief Smart pointer to reference a layer object
 *
 *  See @c OwningLayerPtr
 *
 *  @todo Replace with @c std::observer_ptr<Layer> when supported by
 *  C++ and Cereal.
 */
using ViewingLayerPtr = std::weak_ptr<Layer>;

/** Represents a parallel strategy for a layer. */
struct ParallelStrategy
{
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
  bool enable_subgraph = false;
  /** Branch number in the sub graph. */
  int sub_branch_tag = 0;
  /** percentage of parent resources to be allocated to this branch. */
  int sub_branch_resource_percentage = 0;
  bool operator==(const ParallelStrategy& ps) const
  {
    return sample_groups == ps.sample_groups &&
           sample_splits == ps.sample_splits &&
           depth_groups == ps.depth_groups && depth_splits == ps.depth_splits &&
           height_groups == ps.height_groups &&
           height_splits == ps.height_splits &&
           width_groups == ps.width_groups && width_splits == ps.width_splits &&
           channel_groups == ps.channel_groups &&
           channel_splits == ps.channel_splits &&
           filter_groups == ps.filter_groups &&
           filter_splits == ps.filter_splits &&
           replications == ps.replications &&
           sub_branch_tag == ps.sub_branch_tag &&
           sub_branch_resource_percentage ==
             ps.sub_branch_resource_percentage &&
           enable_subgraph == ps.enable_subgraph;
  }
  bool operator!=(const ParallelStrategy& ps) const { return !(*this == ps); }
};

inline std::ostream& operator<<(std::ostream& os, const ParallelStrategy& ps)
{
  os << "{"
     << "N: " << ps.sample_groups << "/" << ps.sample_splits << ", "
     << "C: " << ps.channel_groups << "/" << ps.channel_splits << ", "
     << "D: " << ps.depth_groups << "/" << ps.depth_splits << ", "
     << "H: " << ps.height_groups << "/" << ps.height_splits << ", "
     << "W: " << ps.width_groups << "/" << ps.width_splits << ", "
     << "F: " << ps.filter_groups << "/" << ps.filter_splits << ", "
     << "R: " << ps.replications << ", "
     << "T: " << ps.sub_branch_tag << ", "
     << "%: " << ps.sub_branch_resource_percentage << ", "
     << "e: " << ps.enable_subgraph << "}";
  return os;
}

inline std::ostream& print_parallel_strategy_header(std::ostream& os)
{
  os << "Axis over which DistConv can parallelize:\n"
     << "\tSamples in the mini-batch (N)\n"
     << "\tDepth, Height, and Width (D x H x W)\n"
     << "\tChannel (C)\n"
     << "\tFilters (F)\n"
     << "\tReplications (R): Number of times the layer is replicated (for FC "
        "layers right now)\n"
     << "\tBranch number in the subgraph (T)\n"
     << "\tPercentage of parent resources to be allocated to this branch (%)\n"
     << "\tEnable subgraph for the layer (e)\n"
     << "\nFor each of the above dimensions there are two fields:\n"
     << "\t# Groups (G): refers to how many reduced-order tensors exist with "
        "respect to that dimension"
     << std::endl
     << "\t             e.g. For a kD tensor you would have a stack of G "
        "(k-1)D tensors"
     << std::endl
     << "\t\t[N, C, D, H, W]" << std::endl
     << "\t\t[2, 1, 4, 1, 1] ---" << std::endl
     << "\t\t                  |" << std::endl
     << "\t\t                  V" << std::endl
     << "\t\t  4 Depth groups: [N, C, H, W]" << std::endl
     << "\t\t                  [2, 1, 1, 1]" << std::endl
     << "\t\t                  [2, 1, 1, 1]" << std::endl
     << "\t\t                  [2, 1, 1, 1]" << std::endl
     << "\t\t                  [2, 1, 1, 1]" << std::endl
     << "\t\t[1, 1, 4, 1, 2] ---" << std::endl
     << "\t\t                  |" << std::endl
     << "\t\t                  V" << std::endl
     << "\t\t 2 Sample groups: [C, D, H, W]" << std::endl
     << "\t\t                  [1, 4, 1, 1]" << std::endl
     << "\t\t                  [1, 4, 1, 1]" << std::endl
     << "\n\tSplit per Dimension (S): Number of groups the dimension is split "
        "over (i.e. split K times) (aka H2 split shape) (must divide groups "
        "evenly).\n"
     << std::endl;

  os << "Reporting order for the parallel strategy" << std::endl;
  os << "{N: G/S"
     << ", C: G/S"
     << ", D: G/S"
     << ", H: G/S"
     << ", W: G/S"
     << ", F: G/S"
     << ", R:"
     << ", T:"
     << ", %:"
     << ", e:"
     << "}";
  return os;
}

enum SubGraphCommunication
{
  PT2PT = 0,
  COLL = 10,
  COLL_OPT = 2
};

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
class Layer
{
  friend class callback::sync_layers;
  friend class callback::memory_profiler;
  friend class KFAC;
  template <hydrogen::Device Device>
  friend class kfac_block_fc_conv;
  template <hydrogen::Device Device>
  friend class kfac_block_channelwise_fc;
  template <hydrogen::Device Device>
  friend class kfac_block_bn;
  template <hydrogen::Device Device>
  friend class kfac_block_gru;

public:
  /** @name Lifecycle */
  ///@{
  Layer();
  virtual ~Layer() = default;

  /** @brief Copy function.
   *  This function dynamically allocates memory for a layer instance
   *  and instantiates a copy. The caller is responsible for
   *  deallocating the instance.
   */
  virtual Layer* copy() const = 0;

  ///@}
  /** @name Metadata modifiers */
  ///@{

  /** @brief Set the layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  void set_name(const std::string name) { m_name = name; }
  /** @brief Set the model that manages this layer. */
  void set_model(model* m) { m_model = m; }

  ///@}
  /** @name Metadata queries */
  ///@{

  /** @brief Get the layer instance's name.
   *
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  std::string get_name() const { return m_name; }

  /** @brief Get a reference to the model that manages this layer.
   *
   *  May be null if this layer is "free" (e.g., during model assembly
   *  or for testing). This will not be null in training applications.
   */
  model* get_model() const noexcept { return m_model; }

  /** @brief Get the layer type's name.
   *
   *  A layer type name should be brief, unique, and human-readable
   *  description of the layer's mathematical operation that is
   *  recognizable to ML practitioners (e.g., "Convolution", "ReLU")
   */
  virtual std::string get_type() const = 0;

  /** @brief Return the layer's input type. */
  virtual std::type_index get_input_datatype() const = 0;

  /** @brief Return the layer's output type. */
  virtual std::type_index get_output_datatype() const = 0;

  /** @brief Get a string representing the layer datatype */
  virtual std::string get_datatype_name() const = 0;

  /** @brief Human-readable description. */
  virtual description get_description() const;

  /** @brief Get data layout of the data tensors.
   *  We assume that the data layouts of the previous activations,
   *  activations, previous error signals, and error signals are the
   *  same. Each concrete layer that is templated on its data layout
   *  should override this function to return its template parameter.
   */
  virtual data_layout get_data_layout() const = 0;
  /** @brief Get the device allocation for the data tensors.
   *  We assume that the decice allocation of the previous activations,
   *  activations, previous error signals, and error signals are the
   *  same. Each concrete layer that is templated on its device allocation
   *  should override this function to return its template parameter.
   */
  virtual El::Device get_device_allocation() const = 0;

  /** @brief Get whether this layer participates on this process.
   *
   *  @note This is technically possible to implement here, but easier
   *        in data_type_layer.
   */
  virtual bool is_participating() const = 0;

  /** @brief Get expected number of parent layers.
   *  A negative value indicates no limit.
   */
  int get_expected_num_parent_layers() const noexcept
  {
    return m_expected_num_parent_layers;
  }

  /** @brief Get expected number of child layers.
   *  A negative value indicates no limit.
   */
  int get_expected_num_child_layers() const noexcept
  {
    return m_expected_num_child_layers;
  }

  /**
   * @brief Returns the necessary tensors for computing backpropagation
   */
  virtual int get_backprop_requirements() const
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS | ACTIVATIONS | WEIGHTS;
  }

  /** @brief Get the parallel strategy for the layer. */
  ParallelStrategy& get_parallel_strategy() noexcept
  {
    return m_parallel_strategy;
  }
  /** @brief Get the parallel strategy for the layer. */
  ParallelStrategy const& get_parallel_strategy() const noexcept
  {
    return m_parallel_strategy;
  }

  ///@}
  /** @name Metadata predicates */
  /**
   * @brief If True, the computation can run in-place (feeding each
   * input activations tensor as the corresponding output activations)
   */
  virtual bool can_run_inplace() const { return false; }

  /** @brief Whether the layer is using a GPU implementation. */
#ifdef LBANN_HAS_GPU
  bool using_gpus() const { return get_device_allocation() == El::Device::GPU; }
#else
  bool using_gpus() const noexcept { return false; }
#endif // LBANN_HAS_GPU

  ///@}
  /** @name Training support */
  ///@{

  /** @brief Forward propagation step.
   *  Apply a mathematical operation to input tensors to obtain output
   *  tensors.
   */
  virtual void forward_prop() = 0;
  /** @brief Backward propagation step.
   *  Given the objective function gradients w.r.t. the output
   *  tensors, compute the gradients w.r.t. the input tensors and
   *  w.r.t. the weights. This is essentially an application of the
   *  chain rule.
   */
  void back_prop();

  /** @brief Update step.
   *  Update the layer's internal members. Note that the optimization
   *  step for the weights happens elsewhere.
   */
  bool update();

  /** @brief Setup layer members
   *
   *  This calls the 'setup_pointers', 'setup_dims', 'setup_matrices',
   *  'setup_data', and 'setup_gpu' (if needed) functions. It is
   *  assumed that pointers to parent/child layers have already been
   *  initialized.
   */
  virtual void setup(size_t max_mini_batch_size,
                     const std::vector<El::Grid*>& grids);

  /** @brief Check that the setup is reasonable. */
  virtual void check_setup();

  ///@}

  /** @brief Write layer to proto file */
  void write_proto(lbann_data::Layer& proto) const;

  /** @name Summarizer support */
  ///@{

  // FIXME (trb 10/03/2023): The lbann_summary class should be
  // reevaluated. This strikes me as a feature that should be moved to
  // a callback and/or replaced by proper performance
  // profilers/counters (e.g., Caliper). More directly: I'm not sure
  // anyone has used the summarizer in at least 2 years and it might
  // be time to trim that out.
  void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step) = 0;

  /** @brief Reset layer stat counters. */
  void reset_counters();

  ///@}
  /** @name Subgraph stuff */
  ///@{

  // (trb 10/03/2023): unused, but accessor is used; kept for symmetry
  void set_communication_flag(SubGraphCommunication type)
  {
    subgraph_communication_method = type;
  }

  // (trb 10/03/2023): used, keeping setter, which is unused.
  SubGraphCommunication get_communication_flag()
  {
    return subgraph_communication_method;
  }

  // (trb 10/03/2023): used
  void set_num_spliting_groups(El::Int spliting_groups)
  {
    m_num_spliting_groups = spliting_groups;
  }

  // (trb 10/03/2023): used
  El::Int get_num_spliting_groups() const { return m_num_spliting_groups; }

  // (trb 10/03/2023): USED BUT THIS IS BAD BECAUSE m_mygrid IS NEVER SET!
  std::shared_ptr<El::Grid> get_mygrid() const
  {
    LBANN_ERROR("This function should not be used.");
    return nullptr;
  }

  // (trb 10/03/2023): used, model.cpp
  void reset_inter_subgrid_vc_comm(std::shared_ptr<El::mpi::Comm> mpi_comm)
  {
    m_interSubGridVCComm = std::move(mpi_comm);
  }

  // (trb 10/03/2023): used
  void set_subgraph_parallelism_execution()
  {
    m_subgraph_parallelism_execution = true;
  }

  // layer-level sub-graph parallelism execution
  // (trb 10/03/2023): used
  bool subgraph_parallelism_execution() const noexcept
  {
    return m_subgraph_parallelism_execution;
  }

  // (trb 10/03/2023): used
  void set_run_layer_in_subgraph() { run_layer_in_subgraph = true; }

  // (trb 10/03/2023): used
  bool get_run_layer_in_subgraph() const noexcept
  {
    return run_layer_in_subgraph;
  }
  ///@}

private:
  /** @brief Add layer specific data to prototext */
  virtual void write_specific_proto(lbann_data::Layer& proto) const = 0;

public:
#ifdef LBANN_HAS_ONNX
  /** @brief Add layer specific data to onnx graph
   *  Fills layer specific data in onnx nodes. Needs to
   *  be overridden by layers that cannot be represented
   *  by a single onnx operator type.
   */
  virtual void fill_onnx_node(onnx::GraphProto& graph) const;

private:
  /** @brief Get ONNX operator type
   *  Unsupported layers and layers that cannot be represented
   *  by a single ONNX operator type will throw an LBANN error.
   *  The operator types for these layers must be included
   *  manually in the overridden fill_onnx_node() function.
   */
  virtual std::string get_onnx_op_type() const;
#endif // LBANN_HAS_ONNX

public:
  /** @name Parent/child accessors */
  ///@{

  const Layer& get_parent_layer(size_t index = 0) const;
  const Layer& get_child_layer(size_t index = 0) const;

  std::vector<const Layer*> get_parent_layers() const;
  std::vector<const Layer*> get_child_layers() const;

  size_t find_parent_layer_index(const Layer& l) const;
  size_t find_child_layer_index(const Layer& l) const;

  /** @brief Get number of parent layers. */
  int get_num_parents() const noexcept { return m_parent_layers.size(); }
  /** @brief Get number of child layers. */
  int get_num_children() const noexcept { return m_child_layers.size(); }

  ///@}
  /** @name Layer pointer manipulation functions */
  ///@{

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
  void clear_parent_layers() { m_parent_layers.clear(); }
  /** @brief Remove pointers to child layers */
  void clear_child_layers() { m_child_layers.clear(); }

  ViewingLayerPtr get_parent_layer_pointer(size_t index) const;
  ViewingLayerPtr get_child_layer_pointer(size_t index) const;

  /** @brief List of pointers to other layers */
  virtual std::vector<ViewingLayerPtr> get_layer_pointers();
  /** @brief Set list of pointers to other layers
   *
   *  Input should match output of @c get_layer_pointers .
   */
  virtual void set_layer_pointers(std::vector<ViewingLayerPtr> layers);

  ///@}
  /** @name Weights access functions */
  ///@{

  /** @brief List of pointers to weights */
  std::vector<ViewingWeightsPtr> get_weights_pointers() const;
  /** @brief Set list of pointers to weights */
  void set_weights_pointers(std::vector<ViewingWeightsPtr> ptrs);

  /** @brief Replace weights with another Layer's weights*/
  void replace_weights(Layer const& other_layer);

  ///@}
  /** @name Tensor access functions */
  ///@{

  /** @brief Get activation tensor corresponding to child layer. */
  virtual const BaseDistMat& get_activations(const Layer& child) const = 0;
  /** @brief Get error signal tensor corresponding to parent layer. */
  virtual const BaseDistMat& get_error_signals(const Layer& parent) const = 0;

  ///@}
  /** @name Tensor dimension access functions */
  ///@{

  /** @brief Get input tensor dimensions. */
  std::vector<int> get_input_dims(size_t input_index = 0) const;
  /** @brief Get input tensor size. */
  int get_input_size(size_t input_index = 0) const;
  /** @brief Get output tensor dimensions. */
  std::vector<int> get_output_dims(size_t output_index = 0) const;
  /** @brief Get output tensor size. */
  int get_output_size(size_t output_index = 0) const;

  /** @brief Set output tensor dimensions. */
  void set_output_dims(std::vector<int> dims, size_t output_index = 0);

  El::Int infer_mini_batch_size_from_parents() const;
  virtual El::Int current_output_mini_batch_size() const = 0;
  virtual El::Int
  infer_mini_batch_size_from_parents_or_default_to_current() const = 0;

  ///@}

  /** Get reference to LBANN communicator. */
  lbann_comm* get_comm() const;

  /** @name Layer parallelism interface */
  ///@{
  /** @brief Get the "layer parallelism" grid tag. */
  int grid_tag() const noexcept;
  /** @brief Set the "layer parallelism" grid tag. */
  void grid_tag(int tag);
  ///@}

  /** @brief Identifying tag for process grid */
  int get_grid_tag() const noexcept;
  /** @brief Set process grid */
  void set_grid_tag(int tag);

  /** @name Hint layer access functions */
  ///@{

  /** @brief Set hint layer.
   *
   *  Properties of the hint layer are used during the setup
   *  phase. For instance, the output tensor dimensions are set to
   *  match the hint layer's first output tensor.
   */
  void set_hint_layer(ViewingLayerPtr l);

  /** @brief Get hint layer. */
  const Layer* get_hint_layer() const;

  ///@}
  /** @name Freeze management functions */
  ///@{

  void freeze();
  void unfreeze();
  bool is_frozen() const;

  ///@}

  /** @brief Set whether to keep or dynamically reallocate error signals.
   *
   *  Passing a value of @c true means to keep the error signals; @c
   *  false means to dynamically reallocate them.
   */
  virtual void set_keep_error_signals(bool) = 0;

  /** @brief If true, the layer will run in-place (the input
   * and output activations point to the same tensor).
   * Value is set during graph setup (in setup_pointers) based
   * on layer traits and neighboring layers.
   */
  bool runs_inplace() const { return m_runs_inplace; }

  /** @brief If true, the layer creates new activations during forward
   * computation and owns their memory. This is used to control freeing
   * activations during model training/evaluation.
   */
  virtual bool owns_activations() const = 0;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** @name Protected lifecycle functions */
  ///@{
  Layer(Layer&& other) = default;
  Layer(Layer const& other);
  Layer& operator=(Layer&& other) = default;
  Layer& operator=(Layer const& other);
  ///@}

  /** @name Weights-related accessors */
  ///@{
  void add_weights(ViewingWeightsPtr w)
  {
    m_weights.emplace_back(std::move(w));
  }
  size_t num_weights() const noexcept { return m_weights.size(); }
  bool has_weights() const noexcept { return num_weights() > 0; }
  bool has_weights(size_t idx) const noexcept
  {
    return ((idx < m_weights.size()) && (!m_weights[idx].expired()));
  }
  void set_num_weights(size_t n) { m_weights.resize(n); }
  void set_weights(size_t idx, ViewingWeightsPtr w)
  {
    m_weights.at(idx) = std::move(w);
  }
  weights const& get_weights(size_t idx) const;

  weights& get_weights(size_t idx);

  void add_as_gradient_source();

  void remove_as_gradient_source();
  ///@}

  // ===========================================================
  // Setup helper functions
  // ===========================================================

  /** @brief Setup process grid
   */
  void setup_grid();
  /** @brief Setup layer pointers.
   *  Called by the 'setup' function. Pointers to parent/child layers
   *  are assumed to be already initialized.
   */
  virtual void setup_pointers();
  /** @brief Setup tensor dimensions
   *  Called by the 'setup' function. If there are any input tensors,
   *  the base method sets all uninitialized output tensor dimensions
   *  equal to the first input tensor dimensions.
   */
  virtual void setup_dims();
  /** @brief Setup buffers for layer inputs and outputs
   *
   *  Called by the 'setup' function. Each column of these distributed
   *  matrices is interpreted as the flattened tensor for a mini-batch
   *  sample. The matrices themselves are constructed by calling the
   *  'construct_matrix' function. If any matrices have already been
   *  setup, they are destroyed and reinstantiated.
   */
  virtual void setup_matrices(const std::vector<El::Grid*>& grids) = 0;
  /** @brief Setup layer data.
   *  Called by the 'setup' function. Memory is allocated for
   *  distributed matrices.
   */
  virtual void setup_data(size_t max_mini_batch_size){};
  /** @brief Setup GPU objects.
   *  Called by the 'setup' function if the layer is on GPUs.
   */
  virtual void setup_gpu() {}

  // ===========================================================
  // Forward prop step helper functions
  // ===========================================================

  /** @brief Setup input tensors.
   *  Called by the 'forward_prop' function. Each input tensor is
   *  setup as a view or copy of the corresponding parent layer's
   *  output tensor.
   */
  virtual void fp_setup_inputs() = 0;
  /** @brief Setup output tensors.
   *  Called by the 'forward_prop' function. Each output tensor is
   *  resized to match the mini-batch size.
   */
  virtual void fp_setup_outputs() = 0;
  /** @brief Apply layer operation.
   *  Called by the 'forward_prop' function. Given the input tensors,
   *  the output tensors are populated with computed values.
   */
  virtual void fp_compute() = 0;

  // ===========================================================
  // Back prop step helper functions
  // ===========================================================

  /** @brief Setup gradient w.r.t. input tensors.
   *  Called by the 'back_prop' function. Each gradient w.r.t. input
   *  tensor is resized to match the mini-batch size.
   */
  virtual void bp_setup_gradient_wrt_inputs() = 0;
  /** @brief Compute objective funciton gradients.
   *  Called by the 'back_prop' function. Given the input, output, and
   *  gradient w.r.t. output tensors, the gradient w.r.t. input
   *  tensors are populated with the computed values and the gradients
   *  w.r.t. the weights are sent to the appropriate optimizers.
   */
  virtual void bp_compute(){};

  // ===========================================================
  // Update step helper functions
  // ===========================================================

  /** @brief Perform the computation for the update step.
   *  Returns false if the layer must reset for a new training epoch.
   */
  virtual bool update_compute() { return true; }

  // ===========================================================
  // Protected class members
  // ===========================================================

  /** Expected number of parent layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_parent_layers = 1;
  /** @brief Expected number of child layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_child_layers = 1;

  /** @brief Reference to model managing this layer. */
  model* m_model = nullptr;

  /** @brief Avoid back prop if frozen */
  bool m_frozen;

  /** @brief Time spent in forward propagation. */
  EvalType m_fp_time;
  /** @brief Time spent in the forward propagation computation. */
  EvalType m_fp_compute_time;
  /** @brief Time spent in backward propagation. */
  EvalType m_bp_time;
  /** @brief Time spent in the backward propagation computation. */
  EvalType m_bp_compute_time;
  /** @brief Time spent in updates. */
  EvalType m_update_time;

  /** @brief Layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  std::string m_name;

  /** @brief If true, the layer will run in-place (the input
   * and output activations point to the same tensor).
   * Value is set during graph setup (in setup_pointers) based
   * on layer traits and neighboring layers.
   */
  bool m_runs_inplace = false;

  /** @name Layer parallelism */
  ///@{

  /** @brief The tag used to choose the grid.
   *
   *  During model setup, this will be checked. If it has not been set
   *  (i.e., it is "-1"), then it will be chosen to match its parents
   *  (which must all be on the same grid -- "transitional" layers
   *  must be explicitly marked).
   *
   *  Temporary: While the legacy "subgraph parallelism"
   *  infrastructure coexists, setup will also check that this
   *  and the subgraph-related "m_grid_tag" are not both set. If using
   *  "subgraph", every layer will leave this as "-1" and the grid
   *  setup will proceed according to the legacy subgraph setup.
   *
   *  After setup, this is guaranteed to be >= 0, except when the
   *  legacy "subgraph" codepath is being used. The actual @c Grid
   *  object can be retrieved through the activation matrices. These
   *  are guaranteed to be assigned the "layer parallelism" grid, if
   *  any.
   */
  int m_lp_grid_tag = -1;

  ///@}

  // -------------------------------------------------------
  // Objects for sub-grid parallelism
  // -------------------------------------------------------
  /// @todo tym: Clean up and document

  /** @brief Identifying tag for process grid
   *
   *  If the tag is negative, the process grid is chosen based on
   *  heuristics. In particular, the layer will attempt to use the
   *  same grid as its parent layers, reverting to the trainer grid if
   *  not possible.
   */
  int m_grid_tag = -1;

  // -------------------------------------------------------
  // Objects from old sub-grid parallelism implementation
  // -------------------------------------------------------
  /// @todo Remove

  SubGraphCommunication subgraph_communication_method = PT2PT;

  // Model-level: Is subgraph parallelism enabled for this Model?
  // Layer-level: Does this layer need subgraph execution (like split and sum
  // layers) Process-level: Does this layer exist in the given process (For e.g.
  // some layers will only run on a subset of processes defined by grid tag)
  // Layer-level sub-graph execution
  bool m_subgraph_parallelism_execution = false;
  // Process-level sub-graph execution
  bool run_layer_in_subgraph = false;

  /** Ranks in grid for the sub-graph */
  std::unique_ptr<std::set<int>> m_subgrid_ranks;

  El::Int m_num_spliting_groups = 1;
  std::shared_ptr<El::mpi::Comm> m_interSubGridVCComm;

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
  friend void attempt_move_error_signal(Layer& parent,
                                        Layer const& child,
                                        std::unique_ptr<BaseDistMat> signal);
  friend void attempt_view_error_signal(Layer& parent,
                                        Layer const& child,
                                        const BaseDistMat& signals);
  friend void deep_copy_error_signal(Layer& parent,
                                     Layer const& child,
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
  virtual void
  view_or_copy_prev_error_signal_(const Layer& child,
                                  const El::BaseDistMatrix& signal) = 0;

  /** @brief Deep-copy the error signals from the specified child
   *         layer.
   *
   *  @param child The layer whence the signal is coming.
   *  @param signal The error signals being sent to this layer.
   */
  virtual void
  deep_copy_prev_error_signal_(const Layer& child,
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
  std::vector<ViewingWeightsPtr> m_weights;

  /** @brief Dimensions of output tensors. */
  std::vector<std::vector<int>> m_output_dims_list;

  /** @brief Hint layer.
   *  During setup, the output tensor dimensions are set to match the
   *  first output tensor of the hint layer. Derived classes may do
   *  more elaborate setup based on the hint layer.
   */
  ViewingLayerPtr m_hint_layer;

  /** @brief Parallel strategy for the layer. */
  ParallelStrategy m_parallel_strategy;

#ifdef LBANN_HAS_DISTCONV
private:
  friend class distconv_adapter;

public:
  /** @brief Indicate whether distconv is enabled. */
  bool distconv_enabled() const;
  /** @brief Indicate whether original input matrices need to be set up. */
  virtual bool keep_original_inputs(int index) const;
  /** @brief Indicate whether original output matrices need to be set up. */
  virtual bool keep_original_outputs(int index) const;
  /** @brief Indicate whether original gradient wrt input matrices need to be
   * set up. */
  virtual bool keep_original_gradient_wrt_inputs(int index) const;
  /** @brief Indicate whether original gradient wrt output matrices need to be
   * set up. */
  virtual bool keep_original_gradient_wrt_outputs(int index) const;
  /** @brief Retrieves distconv adapter. */
  virtual const distconv_adapter& get_distconv_adapter() const;
  /** @brief Retrieves distconv adapter. */
  virtual distconv_adapter& get_distconv_adapter();

protected:
  /** @brief Indicate whether distconv is supported. */
  virtual bool is_distconv_supported() const { return false; }
  /** @brief Pre-initialize distconv attributes needed for setup_data(). */
  void prepare_distconv();
  virtual void setup_distconv_adapter() = 0;
  std::unique_ptr<distconv_adapter>& get_distconv_adapter_ptr()
  {
    return m_dc;
  };
  const std::unique_ptr<distconv_adapter>& get_distconv_adapter_ptr() const
  {
    return m_dc;
  };

private:
  mutable bool m_distconv_enabled = false;
  mutable bool m_distconv_enabled_set = false;
  std::unique_ptr<distconv_adapter> m_dc;
#else
public:
  /** @brief Indicate whether distconv is enabled. */
  bool distconv_enabled() const { return false; }

#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_LAYERS_LAYER_HPP_INCLUDED
