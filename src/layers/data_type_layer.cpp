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

#define LBANN_DATA_TYPE_LAYER_INSTANTIATE

#include "matrix_builder.hpp"

#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/summary_impl.hpp"
#include "lbann/utils/tensor_impl.hpp"
#include "lbann/utils/timer.hpp"

namespace {
template <typename MatrixPtrT>
std::vector<MatrixPtrT> copy_all(std::vector<MatrixPtrT> const& in)
{
  std::vector<MatrixPtrT> out;
  out.reserve(in.size());
  for (auto const& m : in)
    out.emplace_back(m ? m->Copy() : nullptr);
  return out;
}
} // namespace

namespace lbann {

template <typename InputTensorDataType, typename OutputTensorDataType>
data_type_layer<InputTensorDataType, OutputTensorDataType>::data_type_layer(
  data_type_layer const& other)
  : Layer(other), m_persistent_error_signals(other.m_persistent_error_signals)
{

  // Deep matrix copies
  m_inputs = copy_all(other.m_inputs);
  m_outputs = copy_all(other.m_outputs);
  m_gradient_wrt_outputs = copy_all(other.m_gradient_wrt_outputs);
  m_gradient_wrt_inputs = copy_all(other.m_gradient_wrt_inputs);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
data_type_layer<InputTensorDataType, OutputTensorDataType>&
data_type_layer<InputTensorDataType, OutputTensorDataType>::operator=(
  data_type_layer const& other)
{
  Layer::operator=(other);

  // Deep matrix copies
  m_inputs = copy_all(other.m_inputs);
  m_outputs = copy_all(other.m_outputs);
  m_gradient_wrt_outputs = copy_all(other.m_gradient_wrt_outputs);
  m_gradient_wrt_inputs = copy_all(other.m_gradient_wrt_inputs);
  m_persistent_error_signals = other.m_persistent_error_signals;
  return *this;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::forward_prop()
{
  const auto fp_start = get_time();

  // Setup weights proxies
  if (this->has_weights()) {
    if ((m_weights_proxy.size() == 0) || m_weights_proxy[0].empty()) {
      auto const num_weights = this->num_weights();
      m_weights_proxy.resize(num_weights);
      const auto ptrs = this->get_weights_pointers();
      for (size_t ii = 0; ii < num_weights; ++ii) {
        m_weights_proxy[ii].setup(ptrs[ii]);
      }
    }
    for (auto& wp : m_weights_proxy)
      wp.synchronize_with_master();
  }

  // Setup tensors
  const auto& c =
    static_cast<SGDExecutionContext&>(m_model->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) {
    hydrogen::gpu::SynchronizeDevice();
  }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled())
    get_distconv_adapter().fp_setup(mini_batch_size);
#endif // LBANN_HAS_DISTCONV

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled())
    get_distconv_adapter().fp_postprocess();
#endif // LBANN_HAS_DISTCONV

  // Add this layer as a gradient source for weight optimizers
  this->add_as_gradient_source();

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) {
    hydrogen::gpu::SynchronizeDevice();
  }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_fp_time += get_time() - fp_start;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::back_prop_impl_()
{
  const auto bp_start = get_time();

  // Setup tensors
  const auto& c =
    static_cast<SGDExecutionContext&>(m_model->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  bp_setup_gradient_wrt_inputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) {
    hydrogen::gpu::SynchronizeDevice();
  }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled())
    get_distconv_adapter().bp_setup(mini_batch_size);
#endif // LBANN_HAS_DISTCONV

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled())
    get_distconv_adapter().bp_postprocess();
#endif // LBANN_HAS_DISTCONV

  // Remove this layer as a gradient source for weight optimizers
  this->remove_as_gradient_source();

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) {
    hydrogen::gpu::SynchronizeDevice();
  }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_bp_time += get_time() - bp_start;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  summarize_matrices(lbann_summary& summarizer, int step)
{

  // Summarize activation matrices
  const int num_children = get_num_children();
  for (int i = 0; i < num_children; ++i) {
    OutputAbsDistMatReadProxyType<El::Device::CPU> acts(*m_outputs[i]);
    std::string prefix = m_name + "/activations";
    if (num_children > 1) {
      prefix += std::to_string(i);
    }
    summarizer.reduce_mean(prefix + "/mean", acts.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", acts.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", acts.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", acts.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", acts.GetLocked(), step);
  }

  // Summarize error signal matrices
  const int num_parents = get_num_parents();
  for (int i = 0; i < num_parents; ++i) {
    if (!m_gradient_wrt_inputs[i])
      continue;

    InputAbsDistMatReadProxyType<El::Device::CPU> error_signals(
      *m_gradient_wrt_inputs[i]);
    std::string prefix = m_name + "/error_signals";
    if (num_parents > 1) {
      prefix += std::to_string(i);
    }
    summarizer.reduce_mean(prefix + "/mean", error_signals.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", error_signals.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", error_signals.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", error_signals.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2",
                            error_signals.GetLocked(),
                            step);
  }
}

// ===================================================================
// Tensor access functions
// ===================================================================

// Accessing distributed matrices
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_prev_activations(int parent_index) const -> const InputAbsDistMatrixType&
{
  if (parent_index < 0 || parent_index >= (int)m_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_inputs.size() << " previous activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_inputs[parent_index];
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_activations(int child_index)
  const -> const OutputAbsDistMatrixType&
{
  if (child_index < 0 || child_index >= (int)m_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_outputs.size() << " activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_outputs[child_index];
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_prev_error_signals(int child_index) const
  -> const OutputAbsDistMatrixType&
{
  if (child_index < 0 || child_index >= (int)m_gradient_wrt_outputs.size()) {
    LBANN_ERROR("Attempted to access invalid previous error signal matrix "
                "from ",
                m_name,
                ".\n\nRequested index ",
                child_index,
                ", "
                "but there are ",
                m_gradient_wrt_outputs.size(),
                " previous error signal matrices)");
  }
  if (!m_gradient_wrt_outputs[child_index]) {
    LBANN_ERROR("Previous error signal from",
                m_name,
                "(index=",
                child_index,
                ") is not currently allocated.");
  }
  return *m_gradient_wrt_outputs[child_index];
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_error_signals(int parent_index)
  const -> const InputAbsDistMatrixType&
{
  if (parent_index < 0 || parent_index >= (int)m_gradient_wrt_inputs.size()) {
    LBANN_ERROR("Attempted to access invalid error signal matrix "
                "from ",
                m_name,
                ". Requested index ",
                parent_index,
                ", "
                "but there are ",
                m_gradient_wrt_inputs.size(),
                " error signal matrices)");
  }
  if (!m_gradient_wrt_inputs[parent_index]) {
    LBANN_ERROR("Error signal ",
                parent_index,
                " is currently not available.\n",
                "num parents = ",
                get_num_parents(),
                "\n",
                "num children = ",
                get_num_children(),
                "\n");
  }
  return *m_gradient_wrt_inputs[parent_index];
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::get_temp_grad()
  -> OutputAbsDistMatrixType&
{
  return *m_temp_grad[0];
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_branch_tag_input(int tag)
  -> InputAbsDistMatrixType&
{
  if (m_subgrid_tensors_split.size() <= tag)
    LBANN_ERROR("Error Signal Layer Name:", this->get_name(),
                " Layer type:", this->get_type(),
                ". Subgrid branch tag:", tag, 
                " is more than or equal to the split size:", m_subgrid_tensors_split.size(),
                " or subgrid_tensors vector is not initialized properly.");
  return *m_subgrid_tensors_split[tag];
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_branch_tag_input_vector()
  -> std::vector<std::unique_ptr<InputAbsDistMatrixType>>&
{
  return m_subgrid_tensors_split;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_all_activations()
  -> std::vector<std::unique_ptr<OutputAbsDistMatrixType>>&
{
  return m_outputs;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_all_prev_activations()
  -> std::vector<std::unique_ptr<InputAbsDistMatrixType>>&
{
  return m_inputs;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_all_prev_error_signals()
  -> std::vector<std::unique_ptr<OutputAbsDistMatrixType>>&
{
  return m_gradient_wrt_outputs;
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_all_error_signals()
  -> std::vector<std::unique_ptr<InputAbsDistMatrixType>>&
{
  return m_gradient_wrt_inputs;
}

// Accessing non-const distributed matrices
// Note: Using idiom from Item 3, p. 23 in "Effective C++", 3rd ed.,
// by Scott Meyers.
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_activations(int child_index)
  -> OutputAbsDistMatrixType&
{
  return const_cast<OutputAbsDistMatrixType&>(
    static_cast<
      const data_type_layer<InputTensorDataType, OutputTensorDataType>&>(*this)
      .get_activations(child_index));
}

template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_error_signals(int parent_index)
  -> InputAbsDistMatrixType&
{
  return const_cast<InputAbsDistMatrixType&>(
    static_cast<
      const data_type_layer<InputTensorDataType, OutputTensorDataType>&>(*this)
      .get_error_signals(parent_index));
}

// Accessing local matrices
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_local_activations(int child_index) -> OutputAbsMatrixType&
{
  return get_activations(child_index).Matrix();
}
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_local_error_signals(int parent_index) -> InputAbsMatrixType&
{
  return get_error_signals(parent_index).Matrix();
}
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_local_prev_activations(int parent_index) const
  -> const InputAbsMatrixType&
{
  return get_prev_activations(parent_index).LockedMatrix();
}
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_local_activations(int child_index) const -> const OutputAbsMatrixType&
{
  return get_activations(child_index).LockedMatrix();
}
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_local_prev_error_signals(int child_index) const
  -> const OutputAbsMatrixType&
{
  return get_prev_error_signals(child_index).LockedMatrix();
}
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_local_error_signals(int parent_index) const -> const InputAbsMatrixType&
{
  return get_error_signals(parent_index).LockedMatrix();
}

// Accessing matrices corresponding to parent/child layer
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::get_activations(const Layer& child)
  const -> const OutputAbsDistMatrixType&
{
  if (this->get_num_children() <= 0) {
    LBANN_ERROR("This layer has no children");
  }
  const int child_index = find_child_layer_index(child);
  if (child_index >= get_num_children()) {
    std::stringstream err;
    err << "attempted to get activation tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << child.get_name() << "\", "
        << "which is not a child layer";
    LBANN_ERROR(err.str());
  }
  return get_activations(child_index);
}
template <typename InputTensorDataType, typename OutputTensorDataType>
auto data_type_layer<InputTensorDataType, OutputTensorDataType>::
  get_error_signals(const Layer& parent) const -> const InputAbsDistMatrixType&
{
  const int parent_index = find_parent_layer_index(parent);
  if (parent_index >= get_num_parents()) {
    LBANN_ERROR("attempted to get error signal tensor of "
                "layer \"",
                get_name(),
                "\" "
                "corresponding to layer\"",
                parent.get_name(),
                "\", "
                "which is not a parent layer");
  }
  return get_error_signals(parent_index);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::set_keep_error_signals(bool flag)
{
  m_persistent_error_signals = flag;
}

namespace {

// Some indirection around building matrices to keep things tidy in
// the real code. This is just to hide multiple switches without
// building a full-blown dispatch engine... This also keeps bad
// type/device combinations from being instantiated (eg, cpu_fp16 on
// Device::GPU).
using namespace h2::meta;

#ifdef LBANN_HAS_GPU
template <typename T,
          data_layout Layout,
          typename = EnableWhenV<El::IsStorageType<T, El::Device::GPU>>>
auto MakeMatBuilderGPU() -> std::unique_ptr<details::MatrixBuilder<T>>
{
  return std::make_unique<
    details::DefaultMemoryMatrixBuilder<T, Layout, El::Device::GPU>>();
}

template <typename T,
          data_layout Layout,
          typename = EnableUnlessV<El::IsComputeType<T, El::Device::GPU>>,
          typename = void>
auto MakeMatBuilderGPU() -> std::unique_ptr<details::MatrixBuilder<T>>
{
  LBANN_ERROR("Bad type/device combination.");
  return nullptr;
}
#endif // LBANN_HAS_GPU

template <typename T, data_layout Layout>
auto MakeMatBuilderDev(El::Device const device)
  -> std::unique_ptr<details::MatrixBuilder<T>>
{
  switch (device) {
  case El::Device::CPU:
    return std::make_unique<
      details::DefaultMemoryMatrixBuilder<T, Layout, El::Device::CPU>>();
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return MakeMatBuilderGPU<T, Layout>();
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Invalid device type");
  }
}
template <typename T>
auto MakeMatBuilder(data_layout const layout, El::Device const device)
  -> std::unique_ptr<details::MatrixBuilder<T>>
{
  switch (layout) {
  case data_layout::DATA_PARALLEL:
    return MakeMatBuilderDev<T, data_layout::DATA_PARALLEL>(device);
  case data_layout::MODEL_PARALLEL:
    return MakeMatBuilderDev<T, data_layout::MODEL_PARALLEL>(device);
  default:
    LBANN_ERROR("Invalid data layout");
  }
  return nullptr;
}

} // namespace

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::setup_matrices(
  const std::vector<El::Grid*>& grids)
{

  using InputMatrixBuilderType = details::MatrixBuilder<InputTensorDataType>;
  using OutputMatrixBuilderType = details::MatrixBuilder<OutputTensorDataType>;

  // DEBUG
  {
    char* keep_error_signals = getenv("LBANN_KEEP_ERROR_SIGNALS");
    if (!keep_error_signals || (std::stoi(keep_error_signals) == 0))
      m_persistent_error_signals = false;
    else
      m_persistent_error_signals = true;
  }

  // If no CUB, force persistent error signals:
#if defined(HYDROGEN_HAVE_GPU) && !defined(HYDROGEN_HAVE_CUB)
  if (this->get_device_allocation() == El::Device::GPU)
    m_persistent_error_signals = true;
#endif

  // Figure out how to make new matrices
  std::unique_ptr<InputMatrixBuilderType> input_mat_builder =
    MakeMatBuilder<InputTensorDataType>(this->get_data_layout(),
                                        this->get_device_allocation());
  std::unique_ptr<OutputMatrixBuilderType> output_mat_builder =
    MakeMatBuilder<OutputTensorDataType>(this->get_data_layout(),
                                         this->get_device_allocation());

  // Destroy previously setup matrices
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();
  m_temp_grad.clear();
  m_subgrid_tensors_split.clear();

  // Construct matrices
  m_inputs.resize(get_num_parents());
  m_outputs.resize(get_num_children());
  m_gradient_wrt_outputs.resize(get_num_children());
  m_gradient_wrt_inputs.resize(get_num_parents());
  m_temp_grad.resize(1);
  m_subgrid_tensors_split.resize(1);

  // Choose process grid to distribute matrices over
  // int tag = this->get_grid_tag();
  // if (tag < 0) {
  //   // Use tag from parent layers if they are all the same. Otherwise
  //   // use tag 0.
  //   for (int i = 0; i < this->get_num_parents(); ++i) {
  //     auto parent_tag = this->get_parent_layer(i).get_grid_tag();
  //     if (i == 0) {
  //       tag = parent_tag;
  //     }
  //     if (tag != parent_tag) {
  //       tag = -1;
  //       break;
  //     }
  //   }
  //   if (tag < 0) {
  //     tag = 0;
  //   }
  // }
  // if (tag < 0 || tag >= static_cast<int>(grids.size())) {
  //   LBANN_ERROR("attempted to initialize ",
  //               this->get_type(),
  //               " layer \"",
  //               this->get_name(),
  //               "\" ",
  //               "on invalid grid ",
  //               "(grid tag ",
  //               tag,
  //               ", ",
  //               grids.size(),
  //               " grids available)");
  // }
  int tag = this->get_grid_tag();
  // this->reset_mygrid(grids[tag]);
  const El::Grid& grid = *grids[tag];

  if (grid.InGrid())
    this->set_run_layer_in_subgraph();

  auto childs = get_child_layers();
  auto parents = get_parent_layers();

  // Enable Subgraph execution for split layer when one of its
  // child has grid tag greater than 0
  // if (this->get_type() == "split" && 
  //   this->get_child_layers()[0]->get_grid_tag()>0)
  //   this->set_enable_subgraph_variable();

  // if (this->get_parallel_strategy().enable_subgraph)
  //   this->set_subgraph_parallelism_execution();

  if ((this->get_type() == "split" ||
       this->get_type() == "slice") &&
      this->get_model()->is_subgraph_parallelism_enabled() &&
      this->subgraph_parallelism_execution()) {

    // split layer
    m_subgrid_tensors_split.clear();
    m_subgrid_tensors_split.resize(childs[0]->get_num_spliting_groups());

    for (auto& input : m_inputs) {
      input = input_mat_builder->MakeEmpty(grid, 0);
    }

    int count = 0;

    for (auto& output : m_outputs) {
      output = output_mat_builder->MakeEmpty(*grids[childs[count]->get_grid_tag()], 0);
      count++;
    }
    count = 0;
    for (auto& grad_wrt_output : m_gradient_wrt_outputs) {
      grad_wrt_output =
        output_mat_builder->MakeEmpty(*grids[childs[count]->get_grid_tag()], 0);
      count++;
    }

    for (auto& grad_wrt_input : m_gradient_wrt_inputs) {
      grad_wrt_input = input_mat_builder->MakeEmpty(grid, 0);
    }

    for (auto& temp_grad : m_temp_grad) {
      temp_grad = output_mat_builder->MakeEmpty(grid, 0);
    }
    count = 0;
    for (auto& subgrid_tensor : m_subgrid_tensors_split) {
      for (int child_index = 0; child_index < int(childs.size());
           ++child_index) {
        if (childs[child_index]->get_grid_tag() ==
            count + 1) {
          subgrid_tensor =
            output_mat_builder->MakeEmpty(*grids[childs[child_index]->get_grid_tag()],
                                          0);
          count++;
          break;
        }
      }
    }

    // create interprocess subgrid communicator
  }
  else if ((get_type() == "cross_grid_sum" ||
            get_type() == "cross_grid_sum_slice") &&
           this->get_model()->is_subgraph_parallelism_enabled()) {
    m_subgrid_tensors_split.clear();
    m_subgrid_tensors_split.resize(childs[0]->get_num_spliting_groups());

    int count = 0;
    for (auto& input : m_inputs) {
      input = input_mat_builder->MakeEmpty(*grids[parents[count]->get_grid_tag()], 0);
      count++;
    }

    count = 0;

    for (auto& output : m_outputs) {
      output = output_mat_builder->MakeEmpty(*grids[childs[count]->get_grid_tag()], 0);
      count++;
    }
    count = 0;
    for (auto& grad_wrt_output : m_gradient_wrt_outputs) {
      grad_wrt_output =
        output_mat_builder->MakeEmpty(*grids[childs[count]->get_grid_tag()], 0);
      count++;
    }

    count = 0;
    for (auto& grad_wrt_input : m_gradient_wrt_inputs) {
      grad_wrt_input =
        input_mat_builder->MakeEmpty(*grids[parents[count]->get_grid_tag()], 0);
      count++;
    }

    for (auto& temp_grad : m_temp_grad) {
      temp_grad = output_mat_builder->MakeEmpty(grid, 0);
    }
  }
  else if ((get_type() == "sum" || this->get_type() == "concatenate") &&
           this->get_model()->is_subgraph_parallelism_enabled() &&
           this->subgraph_parallelism_execution()) {
    // sum layer

    m_subgrid_tensors_split.clear();
    m_subgrid_tensors_split.resize(this->get_num_spliting_groups());

    int count = 0;
    for (auto& input : m_inputs) {
      input = input_mat_builder->MakeEmpty(*grids[parents[count]->get_grid_tag()], 0);
      count++;
    }

    for (auto& output : m_outputs) {
      output = output_mat_builder->MakeEmpty(grid, 0);
    }

    for (auto& grad_wrt_output : m_gradient_wrt_outputs) {
      grad_wrt_output = output_mat_builder->MakeEmpty(grid, 0);
    }

    count = 0;
    for (auto& grad_wrt_input : m_gradient_wrt_inputs) {
      grad_wrt_input =
        input_mat_builder->MakeEmpty(*grids[parents[count]->get_grid_tag()], 0);
      count++;
    }

    for (auto& temp_grad : m_temp_grad) {
      temp_grad = output_mat_builder->MakeEmpty(grid, 0);
    }

    // auto subgrid_tags = *(this->m_parent_tags);

    count = 1;
    for (auto& subgrid_tensor : m_subgrid_tensors_split) {
      subgrid_tensor =
            input_mat_builder->MakeEmpty(*grids[count],
                                         0);
      count++;
      // for (int parent_index = 0; parent_index < int(parents.size());
      //      ++parent_index) {
      //   if (parents[parent_index]->get_grid_tag() - 1 == count) {
      //     subgrid_tensor =
      //       input_mat_builder->MakeEmpty(*grids[count],
      //                                    0);
      //     count++;
      //     break;
      //   }
      // }
    }
  }
  else {

    for (auto& input : m_inputs) {
      input = input_mat_builder->MakeEmpty(grid, 0);
    }

    for (auto& output : m_outputs) {
      output = output_mat_builder->MakeEmpty(grid, 0);
    }

    for (auto& grad_wrt_output : m_gradient_wrt_outputs) {
      grad_wrt_output = output_mat_builder->MakeEmpty(grid, 0);
    }

    for (auto& grad_wrt_input : m_gradient_wrt_inputs) {
      grad_wrt_input = input_mat_builder->MakeEmpty(grid, 0);
    }

    for (auto& temp_grad : m_temp_grad) {
      temp_grad = output_mat_builder->MakeEmpty(grid, 0);
    }
    for (auto& subgrid_tensor : m_subgrid_tensors_split) {
      subgrid_tensor = output_mat_builder->MakeEmpty(grid, 0);
    }
  }

#ifdef LBANN_HAS_GPU
  // Use directly-allocated GPU memory for forward prop matrices
  // Note: GPU memory pool uses more memory and these buffers are
  // rarely reallocated
  /// @todo Consider using directly-allocated device memory when
  /// training with persistent error signals
  if (this->get_device_allocation() == El::Device::GPU) {
    const auto& arg_parser = global_argument_parser();
    if (!arg_parser.get<bool>(
          LBANN_OPTION_USE_GPU_DEFAULT_MEMORY_IN_FORWARD_PROP)) {
      for (auto& input : m_inputs) {
        input->Matrix().SetMemoryMode(0); // Directly-allocated memory
      }
      for (auto& output : m_outputs) {
        output->Matrix().SetMemoryMode(0); // Directly-allocated memory
      }
    }
  }
#endif // LBANN_HAS_GPU
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::setup_data(
  size_t max_mini_batch_size)
{
  Layer::setup_data(max_mini_batch_size);

  // Initialize input and output tensors
  fp_setup_inputs(max_mini_batch_size);
  fp_setup_outputs(max_mini_batch_size);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::bp_compute()
{
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zero(get_error_signals(i));
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
El::AbstractDistMatrix<InputTensorDataType> const&
data_type_layer<InputTensorDataType, OutputTensorDataType>::weights_values(
  size_t idx) const
{
  if (idx >= m_weights_proxy.size()) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "attempted to access weights ",
                idx,
                ", ",
                "but there are only ",
                m_weights_proxy.size(),
                " weights");
  }
  return m_weights_proxy[idx].values();
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::check_setup()
{
  Layer::check_setup();
  std::stringstream err;

  // Check number of tensors
  const int num_parents = get_num_parents();
  const int num_children = get_num_children();
  if ((int)m_inputs.size() != num_parents ||
      (int)m_outputs.size() != num_children ||
      (int)m_gradient_wrt_outputs.size() != num_children ||
      (int)m_gradient_wrt_inputs.size() != num_parents) {
    err << "layer \"" << get_name() << "\" has an "
        << "invalid number of input and output tensors "
        << "(found " << num_parents << " parent layers, " << num_children
        << " child layers, " << m_inputs.size() << " input tensors, "
        << m_outputs.size() << " output tensors, "
        << m_gradient_wrt_outputs.size() << " gradient w.r.t. output tensors, "
        << m_gradient_wrt_inputs.size() << " gradient w.r.t. input tensors)";
    LBANN_ERROR(err.str());
  }

  // Check that tensors are initialized
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_inputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized input tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_outputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized output tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (!m_gradient_wrt_outputs[i]) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. output tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    if (!m_gradient_wrt_inputs[i]) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. input tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  fp_setup_inputs(El::Int mini_batch_size)
{
  if (get_num_parents() < 1) {
    return;
  }

  // Iterate through input tensors
  for (int i = 0; i < get_num_parents(); ++i) {

#ifdef LBANN_HAS_DISTCONV
    // Skip if tensors are managed by Distconv
    if (!keep_original_inputs(i))
      continue;
#endif // LBANN_HAS_DISTCONV

    // Initialize input tensor
    const auto& parent = get_parent_layer(i);
    const auto& parent_output = parent.get_activations(*this);
    auto& input = *m_inputs[i];
    input.Empty(false);
    if (this->is_subgraph_parallelism_enabled())
    {
      if (get_type()=="sum" or get_type()=="concat")
        input.Resize(input.Width(), mini_batch_size);
    }
    view_or_copy_tensor(parent_output, input);

    // Check input matrix dimensions
    const auto& height = get_input_size(i);
    const auto& width = mini_batch_size;
    if ((input.Height() != height || input.Width() != width)) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "expected an input tensor stored in a " << height << " x " << width
          << " matrix "
          << "from layer \"" << parent.get_name() << "\", but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  fp_setup_outputs(El::Int mini_batch_size)
{
  if (get_num_children() < 1) {
    return;
  }

  // Determine distributed matrix alignment
  const bool align_outputs = get_num_parents() > 0;
  const auto& alignment_dist =
    (align_outputs ? get_prev_activations().DistData()
                   : get_activations().DistData());

  // Initialize output tensors
  for (int i = 0; i < get_num_children(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_outputs(i))
      continue;
#endif // LBANN_HAS_DISTCONV
    auto& output = get_activations(i);
    if (output.Viewing()) {
      LBANN_ERROR(get_name(),
                  " fp_setup_outputs should be overridden",
                  " if it needs to handle outputs that view",
                  " other matrices");
    }
    output.Empty(false);
    if (align_outputs) {
      output.AlignWith(alignment_dist);
    }
    output.Resize(get_output_size(i), mini_batch_size);
  }
}

// Implementation details for back-propagation.
namespace {

// This was just cluttering up things.
void assert_tensor_size(const BaseDistMat& mat,
                        El::Int expected_height,
                        El::Int expected_width,
                        std::string const& this_layer_name,
                        std::string const& child_layer_name)
{
  if ((mat.Height() != expected_height) || (mat.Width() != expected_width)) {
    LBANN_ERROR("layer \"",
                this_layer_name,
                "\" expected a tensor stored in a ",
                expected_height,
                " x ",
                expected_width,
                " matrix from layer "
                "\"",
                child_layer_name,
                "\", but got a ",
                mat.Height(),
                " x ",
                mat.Width(),
                " matrix.");
  }
}

} // namespace

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  view_or_copy_prev_error_signal_(const Layer& child, const BaseDistMat& signal)
{
  auto layer_idx = find_child_layer_index(child);
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_gradient_wrt_outputs(layer_idx))
    return;
#endif // LBANN_HAS_DISTCONV

  // Check the signal size
  assert_tensor_size(signal,
                     get_output_size(layer_idx),
                     m_outputs[layer_idx]->Width(),
                     m_name,
                     child.get_name());

  // If the distributions are compatible, we can just view
  // things. Otherwise, deep-copy the data.
  auto& prev_error_sig = *m_gradient_wrt_outputs[layer_idx];
  view_or_copy_tensor(signal, prev_error_sig);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  move_or_copy_prev_error_signal_(const Layer& child,
                                  std::unique_ptr<BaseDistMat> signal_in)
{
  auto layer_idx = find_child_layer_index(child);
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_gradient_wrt_outputs(layer_idx))
    return;
#endif // LBANN_HAS_DISTCONV

  // Check the signal size
  auto& signal = *signal_in;
  std::cout<<"Layer name:"<<m_name<<" RANK:"<<signal.DistData().grid->VCRank()<<" WRT Out Rank:"<<m_gradient_wrt_outputs[layer_idx]->DistData().grid->VCRank()<<"\n";
  assert_tensor_size(signal,
                     get_output_size(layer_idx),
                     m_outputs[layer_idx]->Width(),
                     m_name,
                     child.get_name());

  // If the distribution is OK, then we can just swap data
  // around. Otherwise, deep copy into correct distribution.
  El::DistData expected_distdata = m_outputs[layer_idx]->DistData();
  if (signal.DistData() == expected_distdata) {
    if (auto sig_ptr =
          dynamic_cast<OutputAbsDistMatrixType*>(signal_in.get())) {
      signal_in.release();
      m_gradient_wrt_outputs[layer_idx].reset(sig_ptr);
    }
    else {
      LBANN_ERROR("Logic error: DistData objects compare equal "
                  "but matrices have different dynamic types.");
    }
  }
  else // Deep copy
  {
    if (!m_gradient_wrt_outputs[layer_idx]) {
      m_gradient_wrt_outputs[layer_idx] =
        MakeMatBuilder<OutputTensorDataType>(this->get_data_layout(),
                                             this->get_device_allocation())
          ->MakeEmpty(*expected_distdata.grid, 0);
    }

    do_tensor_copy(signal, *m_gradient_wrt_outputs[layer_idx]);
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  deep_copy_prev_error_signal_(const Layer& child, const BaseDistMat& signal)
{
  auto layer_idx = find_child_layer_index(child);
#ifdef LBANN_HAS_DISTCONV
  if (!keep_original_gradient_wrt_outputs(layer_idx))
    return;
#endif // LBANN_HAS_DISTCONV

  // Check the signal size
  assert_tensor_size(signal,
                     get_output_size(layer_idx),
                     m_outputs[layer_idx]->Width(),
                     m_name,
                     child.get_name());

  // If the distributions are compatible, we can just view
  // things. Otherwise, deep-copy the data.
  auto& prev_error_sig = *m_gradient_wrt_outputs[layer_idx];
  do_tensor_copy(signal, prev_error_sig);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::clear_prev_error_signals_()
{
  if (!m_persistent_error_signals) {
    for (auto& es : m_gradient_wrt_outputs)
      es->Empty(true);
  }
}

void attempt_view_error_signal(Layer& parent,
                               const Layer& child,
                               const BaseDistMat& signal)
{
  parent.view_or_copy_prev_error_signal_(child, signal);
}

void attempt_move_error_signal(Layer& parent,
                               const Layer& child,
                               std::unique_ptr<BaseDistMat> signal)
{
  parent.move_or_copy_prev_error_signal_(child, std::move(signal));
}

void deep_copy_error_signal(Layer& parent,
                            const Layer& child,
                            const BaseDistMat& signal)
{
  parent.deep_copy_prev_error_signal_(child, signal);
}

// If I have persistent error signals, both my "previous error
// signals" and my new error signals will be persistent. So my parents
// can simply setup views into my error signals, if layout, alignment,
// etc is OK.

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  propagate_error_signals_to_parents_()
{
  for (int i = 0; i < get_num_parents(); ++i) {
    auto& parent = const_cast<Layer&>(get_parent_layer(i));

    // If my error signals persist, my parent can always view them,
    // assuming the distdata is right. Otherwise, my views and my data
    // will be released. Views must be copied and owned data can
    // either be copied or swapped out.
    auto& error_signal = *m_gradient_wrt_inputs[i];
    if (m_persistent_error_signals)
      attempt_view_error_signal(parent, *this, error_signal);
    else if (error_signal.Viewing())
      deep_copy_error_signal(parent, *this, error_signal);
    else
      attempt_move_error_signal(parent,
                                *this,
                                std::move(m_gradient_wrt_inputs[i]));
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType,
                     OutputTensorDataType>::allocate_new_gradients_()
{
  auto parents = get_parent_layers();
  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_gradient_wrt_inputs(i))
      continue;
#endif // LBANN_HAS_DISTCONV
    if (!m_gradient_wrt_inputs[i]) {
      if (get_type() == "sum" &&
          this->subgraph_parallelism_execution()) {
        m_gradient_wrt_inputs[i] =
          MakeMatBuilder<InputTensorDataType>(this->get_data_layout(),
                                              this->get_device_allocation())
            ->MakeEmpty(*(parents[i]->get_mygrid()), 0);
      }
      else {
        m_gradient_wrt_inputs[i] =
          MakeMatBuilder<InputTensorDataType>(this->get_data_layout(),
                                              this->get_device_allocation())
            ->MakeEmpty(m_inputs[i]->Grid(), 0);
      }
    }
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    if ( (get_type() == "sum" or  get_type() == "cross_grid_sum") &&
        this->subgraph_parallelism_execution()) {
      std::cout<<"Running for type:"<<get_type()<<"\n";
    }
    else {
      gradient_wrt_input.AlignWith(get_prev_activations(i));
    }
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  bp_setup_gradient_wrt_inputs(El::Int mini_batch_size)
{
  for (int i = 0; i < get_num_parents(); ++i) {
#ifdef LBANN_HAS_DISTCONV
    if (!keep_original_gradient_wrt_inputs(i))
      continue;
#endif // LBANN_HAS_DISTCONV
    auto& gradient_wrt_input = get_error_signals(i);
    if (gradient_wrt_input.Viewing()) {
      LBANN_ERROR(get_name(),
                  " bp_setup_gradient_wrt_inputs should be overridden",
                  " if it needs to handle error signals that view other",
                  "  matrices");
    }
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
    gradient_wrt_input.Resize(get_input_size(i), mini_batch_size);
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  setup_inter_subgrid_comm_based_on_childs(const El::Grid& grid)
{
  // Now we are creating sub-communicators in model.cpp as this method will lead
  // to several instances of sub-communicators on same rank sets. BUG: NCCL
  // allocates some memory for each communicator, which lead to Out-of-memory
  // (OOM) when we have large number of communicators
  const auto& childs = get_child_layers();

  int indexSubgrid = -1;
  for (int child = 0; child < this->get_num_children(); ++child) {
    if (childs[child]->get_mygrid()->InGrid())

    {
      indexSubgrid = child;
    }
  }
  const int posInSubGrid = childs[indexSubgrid]->get_mygrid()->VCRank();
  const int posInGrid = grid.ViewingRank();
  auto& interSubgridComm = this->get_subgrid_comm();
  El::mpi::Split(this->get_comm()->get_trainer_comm(),
                 posInSubGrid,
                 posInGrid,
                 interSubgridComm);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  setup_inter_subgrid_comm_based_on_parents(const El::Grid& grid)
{
  // Now we are creating sub-communicators in model.cpp as this method will lead
  // to several instances of sub-communicators on same rank sets. BUG: NCCL
  // allocates some memory for each communicator, which lead to Out-of-memory
  // (OOM) when we have large number of communicators

  const auto& parents = get_parent_layers();

  int indexSubgrid = -1;
  for (int parent = 0; parent < this->get_num_parents(); ++parent) {
    if (parents[parent]->get_mygrid()->InGrid()) {
      indexSubgrid = parent;
    }
  }
  const int posInSubGrid = parents[indexSubgrid]->get_mygrid()->VCRank();
  const int posInGrid = grid.ViewingRank();
  auto& interSubgridComm = this->get_subgrid_comm();
  El::mpi::Split(this->get_comm()->get_trainer_comm(),
                 posInSubGrid,
                 posInGrid,
                 interSubgridComm);
}

#ifdef LBANN_HAS_DISTCONV
template <typename InputTensorDataType, typename OutputTensorDataType>
void data_type_layer<InputTensorDataType, OutputTensorDataType>::
  setup_distconv_adapter(const DataReaderMetaData& dr_metadata)
{
  this->get_distconv_adapter_ptr() = std::make_unique<
    data_type_distconv_adapter<InputTensorDataType, OutputTensorDataType>>(
    *this);
}

template <typename InputTensorDataType, typename OutputTensorDataType>
data_type_distconv_adapter<InputTensorDataType, OutputTensorDataType>&
data_type_layer<InputTensorDataType,
                OutputTensorDataType>::get_distconv_adapter()
{
  return const_cast<
    data_type_distconv_adapter<InputTensorDataType, OutputTensorDataType>&>(
    static_cast<
      const data_type_layer<InputTensorDataType, OutputTensorDataType>&>(*this)
      .get_distconv_adapter());
}

template <typename InputTensorDataType, typename OutputTensorDataType>
const data_type_distconv_adapter<InputTensorDataType, OutputTensorDataType>&
data_type_layer<InputTensorDataType,
                OutputTensorDataType>::get_distconv_adapter() const
{
  return dynamic_cast<const data_type_distconv_adapter<InputTensorDataType,
                                                       OutputTensorDataType>&>(
    *get_distconv_adapter_ptr());
}
#endif // LBANN_HAS_DISTCONV

#define PROTO(T) template class data_type_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
