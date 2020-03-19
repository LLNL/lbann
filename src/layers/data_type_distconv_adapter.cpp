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

#include "lbann/layers/data_type_distconv_adapter.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

namespace lbann {

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_activations(const Layer& child) const {
  if (layer().get_num_children() == 0) {
    LBANN_ERROR("This layer has no children");
  }
  const int child_index = layer().find_child_layer_index(&child);
  if (child_index >= layer().get_num_children()) {
    LBANN_ERROR("attempted to get activation tensor of ",
                "layer \"", get_name(), "\" ",
                "corresponding to layer\"", child.get_name(), "\", ",
                "which is not a child layer");
  }
  return get_activations(child_index);
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_activations(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_outputs.size()) {
    LBANN_ERROR("attempted to access invalid distconv activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", child_index, ", but there are ",
                m_outputs.size(), " activation tensors)");
  }
  const auto &tensor_ptr = m_outputs[child_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("activation tensor of layer ", get_name(),
                " is not set at index ", child_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_activations(int child_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(*this).get_activations(child_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_activations(
    int child_index) const {
  if (child_index < 0 || child_index >= (int) m_original_outputs.size()) {
    LBANN_ERROR("attempted to access invalid original activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", child_index, ", but there are ",
                m_original_outputs.size(), " original activation tensors)");
  }
  const auto &tensor_ptr = m_original_outputs[child_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("original activation tensor of layer ", get_name(),
                " is not set at index ", child_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_activations(
    int child_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_original_activations(child_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_prev_activations(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_inputs.size()) {
    LBANN_ERROR("attempted to access invalid distconv previous activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", parent_index, ", but there are ",
                m_inputs.size(), " previous activation tensors)");
  }
  const auto &tensor_ptr = m_inputs[parent_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("previous activation tensor of layer ", get_name(),
                " is not set at index ", parent_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_prev_activations(int parent_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_prev_activations(parent_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_prev_activations(
    int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_original_inputs.size()) {
    LBANN_ERROR("attempted to access invalid original previous activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", parent_index, ", but there are ",
                m_original_inputs.size(), " original previous activation tensors)");
  }
  const auto &tensor_ptr = m_original_inputs[parent_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("original previous activation tensor of layer ", get_name(),
                " is not set at index ", parent_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_prev_activations(
    int parent_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_original_prev_activations(parent_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_error_signals(const Layer& parent) const {
  if (layer().get_num_parents() == 0) {
    LBANN_ERROR("This layer has no parents");
  }
  const int parent_index = layer().find_parent_layer_index(&parent);
  if (parent_index >= layer().get_num_parents()) {
    LBANN_ERROR("attempted to get error signal tensor of ",
                "layer \"", get_name(), "\" ",
                "corresponding to layer\"", parent.get_name(), "\", ",
                "which is not a parent layer");
  }
  return get_error_signals(parent_index);
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_error_signals(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_gradient_wrt_inputs.size()) {
    LBANN_ERROR("attempted to access invalid distconv error signal tensor ",
                "from ", get_name(), " ",
                "(requested index ", parent_index, ", but there are ",
                m_gradient_wrt_inputs.size(), " error signal tensors)");
  }
  const auto &tensor_ptr = m_gradient_wrt_inputs[parent_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("error signal tensor of layer ", get_name(),
                " is not set at index ", parent_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_error_signals(int parent_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_error_signals(parent_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_error_signals(
    int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_original_gradient_wrt_inputs.size()) {
    LBANN_ERROR("attempted to access invalid original error signal tensor ",
                "from ", get_name(), " ",
                "(requested index ", parent_index, ", but there are ",
                m_original_gradient_wrt_inputs.size(), " original error signal tensors)");
  }
  const auto &tensor_ptr = m_original_gradient_wrt_inputs[parent_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("original error signal tensor of layer ", get_name(),
                " is not set at index ", parent_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_error_signals(
    int parent_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_original_error_signals(parent_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_prev_error_signals(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_gradient_wrt_outputs.size()) {
    LBANN_ERROR("attempted to access invalid distconv previous error signal tensor ",
                "from ", get_name(), " ",
                "(requested index ", child_index, ", but there are ",
                m_gradient_wrt_outputs.size(), " previous error signal tensors)");
  }
  const auto &tensor_ptr = m_gradient_wrt_outputs[child_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("previous error signal tensor of layer ", get_name(),
                " is not set at index ", child_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_prev_error_signals(int child_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_prev_error_signals(child_index));
}

template <typename TensorDataType>
const typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_prev_error_signals(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_original_gradient_wrt_outputs.size()) {
    LBANN_ERROR("attempted to access invalid original previous error signal tensor ",
                "from ", get_name(), " ",
                "(requested index ", child_index, ", but there are ",
                m_original_gradient_wrt_outputs.size(), " previous error signal tensors)");
  }
  const auto &tensor_ptr = m_original_gradient_wrt_outputs[child_index];
  if (tensor_ptr == nullptr) {
    LBANN_ERROR("original previous error signal tensor of layer ", get_name(),
                " is not set at index ", child_index);
  }
  return *tensor_ptr;
}

template <typename TensorDataType>
typename data_type_distconv_adapter<TensorDataType>::TensorDevType&
data_type_distconv_adapter<TensorDataType>::get_original_prev_error_signals(int child_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_adapter<TensorDataType>&>(
          *this).get_original_prev_error_signals(child_index));
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_prev_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const auto input_tensor_shape = get_prev_activations_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(
      l.get_num_dims());
  auto input_local_shape = input_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  input_local_shape[-1] = 0;

  m_original_inputs.clear();
  m_original_inputs.resize(l.get_num_parents());

  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (parent_copy_in_required(i)) {
      m_original_inputs[i] = make_unique<TensorDevType>(
          input_tensor_shape, loc, sample_dist, input_local_shape);
    } else if (parent_shuffle_required(i)) {
      // NOTE: previous activations are assumed to be of the same
      // tensor data type.
      // Create a shallow copy of the activations of the prev layer
      const auto &parent_activations =
          dynamic_cast<const TensorDevType&>(
              l.get_parent_layers()[i]->dc().get_activations(l));
      m_original_inputs[i] = make_unique<TensorDevType>(
          parent_activations);
    }
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_prev_activations(
    const dc::Dist& dist) {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const auto shape = get_prev_activations_shape();
  const auto local_shape = get_prev_activations_local_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (parent_copy_in_required(i) || parent_shuffle_required(i)) {
      if (i != 0) LBANN_ERROR("Copyin of non-first tensor not supported yet");
      m_inputs.emplace_back(make_unique<TensorDevType>(
          shape, loc, dist, local_shape));
      assert0(m_inputs.back()->allocate());
      m_inputs.back()->zero(dc::get_stream());
    } else {
      // Create a shallow copy
      const auto &parent_activations =
          dynamic_cast<const TensorDevType&>(
              l.get_parent_layers()[i]->dc().get_activations(l));
      // Sanity check
      assert_always(parent_activations.get_distribution() == dist);
      m_inputs.emplace_back(make_unique<TensorDevType>(parent_activations));
    }
  }

  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "prev activations: " << get_prev_activations();
}


template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_prev_activations_shape(
    int input_index) const {
  const auto input_dims = layer().get_input_dims(input_index);
  std::vector<int> input_tensor_shape_v(input_dims.rbegin(), input_dims.rend());
  input_tensor_shape_v.push_back(layer().get_model()->get_max_mini_batch_size());
  return dc::Shape(input_tensor_shape_v);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_prev_activations_local_shape(
    int input_index) const {
  // No enforced local shape.
  return dc::Shape(this->get_num_dims(), 0);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_activations_shape(
    int output_index) const {
  const auto output_dims = layer().get_output_dims(output_index);
  std::vector<int> output_tensor_shape_v(output_dims.rbegin(), output_dims.rend());
  output_tensor_shape_v.push_back(layer().get_model()->get_max_mini_batch_size());
  return dc::Shape(output_tensor_shape_v);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_activations_local_shape(int index) const {
  // Note that, as the default case, it is assumed that the local
  // shape is the same as the local shape of the first previous
  // activations.
  if (index > 0) {
    LBANN_ERROR("Unknown local shape for activations[", index, "]");
  }
  return get_prev_activations(0).get_local_shape();
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_prev_error_signals_shape(
    int index) const {
  // Activations and previous error signals should have the same shape.
  return get_activations_shape(index);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_prev_error_signals_local_shape(int index) const {
  // Activations and previous error signals should have the same local
  // shape.
  return get_activations_local_shape(index);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_error_signals_shape(
    int index) const {
  // Previous activations and error signals should have the same shape.
  return get_prev_activations_shape(index);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_error_signals_local_shape(int index) const {
  // Previous activations and error signals should have the same local
  // shape.
  return get_prev_activations(index).get_local_shape();
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_activations(
    const dc::Dist& dist) {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const dc::Shape output_tensor_shape = get_activations_shape();
  const auto activations_local_shape =
      get_activations_local_shape();
  m_outputs.emplace_back(make_unique<TensorDevType>(
      output_tensor_shape,
      loc, dist, activations_local_shape));
  assert0(m_outputs.back()->allocate());
  m_outputs.back()->zero(dc::get_stream());
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(l.get_num_dims());
  const auto output_tensor_shape = get_activations_shape();
  assert_always(!output_tensor_shape.is_empty());
  auto output_local_shape = output_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  output_local_shape[-1] = 0;

  m_original_outputs.clear();
  m_original_outputs.resize(l.get_num_children());

  // Create a original tensor only when copyout is needed. Note that
  // when the next layer is a distconv layer and has a different
  // distribution, tensor shuffling is necessary but is done at the
  // next layer.
  for (int i = 0; i < l.get_num_children(); ++i) {
    if (!child_copy_out_required(i)) continue;
    m_original_outputs[i] = make_unique<TensorDevType>(
        output_tensor_shape, loc, sample_dist, output_local_shape);
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_prev_error_signals(
    const dc::Dist& dist) {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const auto shape = get_prev_error_signals_shape();
  const auto local_shape = get_prev_error_signals_local_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  for (int i = 0; i < l.get_num_children(); ++i) {
    if (child_copy_out_required(i) || child_shuffle_required(i)) {
      m_gradient_wrt_outputs.emplace_back(make_unique<TensorDevType>(
          shape, loc, dist, local_shape));
      assert0(m_gradient_wrt_outputs.back()->allocate());
      m_gradient_wrt_outputs.back()->zero(dc::get_stream());
    } else {
      // Create a shallow copy
      const auto &child_error_signals =
          dynamic_cast<const TensorDevType&>(
              l.get_child_layers()[i]->dc().get_error_signals(l));
      // Just sanity check
      assert_always(child_error_signals.get_distribution() == dist);
      m_gradient_wrt_outputs.emplace_back(make_unique<TensorDevType>(
          child_error_signals));
    }
  }
  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "prev error signals: " << get_prev_error_signals();
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_prev_error_signals() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const auto shape = get_prev_error_signals_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto dist = dc::get_hydrogen_data_parallel_distribution(
      l.get_num_dims());
  auto local_shape = shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  local_shape[-1] = 0;

  m_original_gradient_wrt_outputs.clear();
  m_original_gradient_wrt_outputs.resize(l.get_num_parents());

  for (int i = 0; i < l.get_num_children(); ++i) {
    if (this->child_copy_out_required(i)) {
      m_original_gradient_wrt_outputs[i] = make_unique<TensorDevType>(
          shape, loc, dist, local_shape);
    } else if (this->child_shuffle_required(i)) {
      // NOTE: previous activations are assumed to be of the same
      // tensor data type.
      // Create a shallow copy of the activations of the prev layer
      const auto &child_error_signals =
          dynamic_cast<const TensorDevType&>(
              l.get_child_layers()[i]->dc().get_error_signals(l));
      m_original_gradient_wrt_outputs[i] = make_unique<TensorDevType>(
          child_error_signals);
    }
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_error_signals(
    const dc::Dist& dist) {
  const auto shape = get_error_signals_shape();
  const auto local_shape = get_error_signals_local_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  m_gradient_wrt_inputs.emplace_back(make_unique<TensorDevType>(
      shape, loc, dist, local_shape));
  if (layer().skip_first_layer_bp()) {
    dc::MPIPrintStreamDebug()
        << get_name() << ": skipping allocation of error signals";
  } else {
    assert0(m_gradient_wrt_inputs.back()->allocate());
    m_gradient_wrt_inputs.back()->zero(dc::get_stream());
  }
  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "error signals: " << get_error_signals();
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_error_signals() {
  const auto shape = get_error_signals_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto dist = dc::get_hydrogen_data_parallel_distribution(
      get_num_dims());
  auto local_shape = shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  local_shape[-1] = 0;

  // TODO: Only the first error signal tensor is handled
  m_original_gradient_wrt_inputs.emplace_back(make_unique<TensorDevType>(
      shape, loc, dist, local_shape));
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::
set_original_activations_outermost_dimension(size_t dim) {
  for (auto &t: m_original_outputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
}

namespace {
template <typename TensorDataType>
dc::TensorShuffler<TensorDataType> &get_shuffler(
    const Layer &layer,
    std::array<dc::TensorShuffler<TensorDataType>*, 4> &shufflers,
    const dc::TensorDev<TensorDataType> &src,
    const dc::TensorDev<TensorDataType> &dst) {
  const auto& c = static_cast<sgd_execution_context&>(
      layer.get_model()->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  int shuffler_idx = -1;
  if (layer.get_model()->get_max_mini_batch_size() == mini_batch_size) {
    shuffler_idx = 0;
  } else {
    // The last remaining mini-batches for the train, validation, and
    // testing modes
    auto mode = layer.get_model()->get_execution_context().get_execution_mode();
    auto ctxt_idx = static_cast<int>(mode);
    assert_always(ctxt_idx >= 0 && ctxt_idx < 3);
    shuffler_idx = ctxt_idx + 1;
  }
  assert_always(shuffler_idx >= 0 && shuffler_idx < 4);
  if (shufflers[shuffler_idx] == nullptr) {
    shufflers[shuffler_idx] = dc::get_tensor_shuffler(src, dst);
  }
  return *shufflers[shuffler_idx];
}
} // namespace

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_prev_activations_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_prev_activations_shufflers, src, dst);
}

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_activations_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_activations_shufflers, src, dst);
}

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_prev_error_signals_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_prev_error_signals_shufflers, src, dst);
}

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_error_signals_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_error_signals_shufflers, src, dst);
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::ensure_prev_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (!(parent_copy_in_required(i) || parent_shuffle_required(i))) {
      continue;
    }
    if (i != 0) {
      LBANN_ERROR("Distconv assumes non-first tensors are available as distconv tensors");
    }
    if (parent_copy_in_required(i)) {
      dc::MPIPrintStreamDebug()
          << "Copying previous activations from sample decomposition";
      assert0(dc::tensor::View(
          get_original_prev_activations(),
          l.get_prev_activations().LockedBuffer()));
    }
    auto &shuffler = get_prev_activations_shuffler(
        get_original_prev_activations(),
        get_prev_activations());
    shuffler.shuffle_forward(
        get_original_prev_activations().get_const_base_ptr(),
        get_prev_activations().get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::copy_out_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_children(); ++i) {
    if (!child_copy_out_required(i)) continue;

    if (i != 0) LBANN_ERROR("Copyout of non-first tensor not supported");

    dc::MPIPrintStreamDebug()
        << "Copying activations back to sample decomposition";
    assert0(dc::tensor::View(
        get_original_activations(), l.get_activations().Buffer()));
    auto &shuffler = get_activations_shuffler(
        get_activations(),
        get_original_activations());
    shuffler.shuffle_forward(
        get_activations().get_const_base_ptr(),
        get_original_activations().get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::ensure_prev_error_signals() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_children(); ++i) {
    if (!(child_copy_out_required(i) || child_shuffle_required(i))) {
      continue;
    }
    if (i != 0) {
      LBANN_ERROR("Distconv assumes non-first tensors are available as distconv tensors");
    }
    if (child_copy_out_required(i)) {
      dc::MPIPrintStreamDebug()
          << "Copying previous error signals from sample decomposition";
      assert0(dc::tensor::View(
          get_original_prev_error_signals(i),
          l.get_prev_error_signals(i).LockedBuffer()));
    }
    auto &shuffler = get_prev_error_signals_shuffler(
        get_original_prev_error_signals(i),
        get_prev_error_signals(i));
    shuffler.shuffle_forward(
        get_original_prev_error_signals(i).get_const_base_ptr(),
        get_prev_error_signals(i).get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::copy_out_error_signals() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());

  if (l.skip_first_layer_bp()) {
    // No need to copy back when the parent is an input layer
    dc::MPIPrintStreamDebug()
        << "Skipping copy back as this layer is the first layer";
    return;
  }

  // No need to copy back as the original layer compute function
  // will be called
  if (l.get_exit_count() == 0) return;

  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (!parent_copy_in_required(i)) continue;

    if (i != 0) LBANN_ERROR("Copyout of non-first tensor not supported");

    dc::MPIPrintStreamDebug()
        << "Copying error signals back to sample decomposition";
    assert0(dc::tensor::View(
        get_original_error_signals(i),
        l.get_error_signals(i).Buffer()));
    auto &shuffler = get_error_signals_shuffler(
        get_error_signals(i),
        get_original_error_signals(i));
    shuffler.shuffle_forward(
        get_error_signals(i).get_const_base_ptr(),
        get_original_error_signals(i).get_base_ptr(),
        El::GPUManager::Stream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_activations() const {
  const auto &l = dynamic_cast<const data_type_layer<TensorDataType>&>(layer());
  dc::dump_tensor(l.early_terminate_last_iteration(),
                  get_activations(), get_name() + "_activations");
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_original_activations() {
  const auto &l = dynamic_cast<const data_type_layer<TensorDataType>&>(layer());
  assert0(dc::tensor::View(
      get_original_activations(), l.get_activations().LockedBuffer()));
  dc::dump_tensor(l.early_terminate_last_iteration(),
                  get_original_activations(),
                  get_name() + "_activations_original");
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_error_signals() const {
  const auto &l = dynamic_cast<const data_type_layer<TensorDataType>&>(layer());
  dc::dump_tensor(l.early_terminate_last_iteration(),
                  get_error_signals(0), get_name() + "_error_signals");
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_original_error_signals() {
  const auto &l = dynamic_cast<const data_type_layer<TensorDataType>&>(layer());
  assert0(dc::tensor::View(
      get_original_activations(), l.get_activations().LockedBuffer()));
  dc::dump_tensor(l.early_terminate_last_iteration(),
                  get_original_error_signals(0),
                  get_name() +  "_error_signals_original");
}

#define PROTO(T)                                \
  template class data_type_distconv_adapter<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
