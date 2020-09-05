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
#include "lbann/trainers/trainer.hpp"

namespace lbann {

namespace {
template <typename TensorDataType>
using TensorDevPtr = std::unique_ptr<typename data_type_distconv_adapter<TensorDataType>::TensorDevType>;
} // namespace

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
  m_original_inputs.clear();
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    m_original_inputs.emplace_back(setup_original_prev_activations_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_original_prev_activations_i(int index) const {
  TensorDevPtr<TensorDataType> t = nullptr;
  if (parent_copy_required(index)) {
    const auto shape = get_prev_activations_shape();
    auto local_shape = shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[-1] = 0;
    const auto dist = dc::get_hydrogen_data_parallel_distribution(
        dc::get_num_dims(layer()));
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
  } else if (parent_shuffle_required(index)) {
    // NOTE: previous activations are assumed to be of the same
    // tensor data type.
    // Create a shallow copy of the activations of the prev layer
    const auto &parent_activations =
        dynamic_cast<const TensorDevType&>(
            layer().get_parent_layers()[index]->get_distconv_adapter().get_activations(layer()));
    t = make_unique<TensorDevType>(parent_activations);
  }
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_prev_activations() {
  m_inputs.clear();
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    m_inputs.emplace_back(setup_prev_activations_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_prev_activations_i(int index) const {
  const auto &dist = this->get_prev_activations_dist();
  TensorDevPtr<TensorDataType> t = nullptr;
  if (parent_copy_required(index) || parent_shuffle_required(index)) {
    if (index != 0) LBANN_ERROR("Copyin of non-first tensor not supported yet");
    const auto shape = get_prev_activations_shape(index);
    const auto local_shape = get_prev_activations_local_shape(index);
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
    assert0(t->allocate());
    t->zero(hydrogen::cuda::GetDefaultStream());
  } else {
    // Create a shallow copy
    const auto &parent_activations =
        dynamic_cast<const TensorDevType&>(
            layer().get_parent_layers()[index]->get_distconv_adapter().get_activations(layer()));
    // Sanity check
    assert_always(parent_activations.get_distribution() == dist);
    t = make_unique<TensorDevType>(parent_activations);
  }
  return t;
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_prev_activations_shape(
    int input_index) const {
  const auto input_dims = layer().get_input_dims(input_index);
  std::vector<int> input_tensor_shape_v(input_dims.rbegin(), input_dims.rend());
  input_tensor_shape_v.push_back(get_max_mini_batch_size());
  return dc::Shape(input_tensor_shape_v);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_prev_activations_local_shape(
    int input_index) const {
  // No enforced local shape.
  return dc::Shape(dc::get_num_dims(layer()), 0);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_adapter<TensorDataType>::get_activations_shape(
    int output_index) const {
  const auto output_dims = layer().get_output_dims(output_index);
  std::vector<int> output_tensor_shape_v(output_dims.rbegin(), output_dims.rend());
  output_tensor_shape_v.push_back(get_max_mini_batch_size());
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
void data_type_distconv_adapter<TensorDataType>::setup_activations() {
  m_outputs.clear();
  for (int i = 0; i < layer().get_num_children(); ++i) {
    m_outputs.emplace_back(setup_activations_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_activations_i(int index) const {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto &dist = this->get_activations_dist();
  const auto shape = get_activations_shape(index);
  const auto local_shape = get_activations_local_shape(index);
  auto t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
  assert0(t->allocate());
  t->zero(hydrogen::cuda::GetDefaultStream());
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_activations() {
  m_original_outputs.clear();
  for (int i = 0; i < layer().get_num_children(); ++i) {
    m_original_outputs.emplace_back(setup_original_activations_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_original_activations_i(int index) const {
  // Create a original tensor only when copyout is needed. Note that
  // when the next layer is a distconv layer and has a different
  // distribution, tensor shuffling is necessary but is done at the
  // next layer.
  TensorDevPtr<TensorDataType> t = nullptr;
  if (child_copy_required(index)) {
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    const auto dist = dc::get_hydrogen_data_parallel_distribution(dc::get_num_dims(layer()));
    const auto shape = get_activations_shape(index);
    assert_always(!shape.is_empty());
    auto local_shape = shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[-1] = 0;
    t = make_unique<TensorDevType>(
        shape, loc, dist, local_shape);
  }
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_prev_error_signals() {
  m_gradient_wrt_outputs.clear();
  for (int i = 0; i < layer().get_num_children(); ++i) {
    m_gradient_wrt_outputs.emplace_back(setup_prev_error_signals_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_prev_error_signals_i(int index) const {
  TensorDevPtr<TensorDataType> t = nullptr;
  const auto &dist = this->get_prev_error_signals_dist();
  if (child_copy_required(index) || child_shuffle_required(index)) {
    const auto shape = get_prev_error_signals_shape(index);
    const auto local_shape = get_prev_error_signals_local_shape(index);
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
    assert0(t->allocate());
    t->zero(hydrogen::cuda::GetDefaultStream());
  } else {
    // Create a shallow copy
    const auto &child_error_signals =
        dynamic_cast<const TensorDevType&>(
            layer().get_child_layers()[index]->get_distconv_adapter().get_error_signals(layer()));
    // Just sanity check
    assert_always(child_error_signals.get_distribution() == dist);
    t = make_unique<TensorDevType>(child_error_signals);
  }
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_prev_error_signals() {
 m_original_gradient_wrt_outputs.clear();
 for (int i = 0; i < layer().get_num_children(); ++i) {
   m_original_gradient_wrt_outputs.emplace_back(
       setup_original_prev_error_signals_i(i));
 }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_original_prev_error_signals_i(int index) const {
  TensorDevPtr<TensorDataType> t = nullptr;
  if (this->child_copy_required(index)) {
    const auto shape = get_prev_error_signals_shape(index);
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    const auto dist = dc::get_hydrogen_data_parallel_distribution(
        dc::get_num_dims(layer()));
    auto local_shape = shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[-1] = 0;
    t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
  } else if (this->child_shuffle_required(index)) {
    // NOTE: previous activations are assumed to be of the same
    // tensor data type.
    // Create a shallow copy of the activations of the prev layer
    const auto &child_error_signals =
        dynamic_cast<const TensorDevType&>(
            layer().get_child_layers()[index]->get_distconv_adapter().get_error_signals(layer()));
    t = make_unique<TensorDevType>(child_error_signals);
  }
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_error_signals() {
  m_gradient_wrt_inputs.clear();
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    m_gradient_wrt_inputs.emplace_back(setup_error_signals_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_error_signals_i(int index) const {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto &dist = this->get_error_signals_dist();
  const auto shape = get_error_signals_shape(index);
  const auto local_shape = get_error_signals_local_shape(index);
  auto t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
  assert0(t->allocate());
  t->zero(hydrogen::cuda::GetDefaultStream());
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::setup_original_error_signals() {
  m_original_gradient_wrt_inputs.clear();
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    m_original_gradient_wrt_inputs.emplace_back(
        setup_original_error_signals_i(i));
  }
}

template <typename TensorDataType>
TensorDevPtr<TensorDataType> data_type_distconv_adapter<TensorDataType>::
setup_original_error_signals_i(int index) const {
  TensorDevPtr<TensorDataType> t = nullptr;
  if (parent_copy_required(index)) {
    const auto shape = get_error_signals_shape(index);
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    const auto dist = dc::get_hydrogen_data_parallel_distribution(
        dc::get_num_dims(layer()));
    auto local_shape = shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[-1] = 0;
    t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
  }
  return t;
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::
set_activations_outermost_dimension(size_t dim) {
  for (auto &t: m_inputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
  for (auto &t: m_original_inputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
  for (auto &t: m_outputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
  for (auto &t: m_original_outputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::
set_error_signals_outermost_dimension(size_t dim) {
  for (auto &t: m_gradient_wrt_outputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
  for (auto &t: m_original_gradient_wrt_outputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
  for (auto &t: m_gradient_wrt_inputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
  for (auto &t: m_original_gradient_wrt_inputs) {
    if (t == nullptr) continue;
    t->set_outermost_dimension(dim);
    assert_eq(t->get_shape()[-1], dim);
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::fp_setup(El::Int mini_batch_size) {
  const auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  set_activations_outermost_dimension(mini_batch_size);
  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (parent_copy_required(i) || parent_shuffle_required(i)) {
      if (i != 0) {
        LBANN_ERROR("Copyin non-first tensor not supported");
      }
      if (parent_copy_required(i)) {
        // Parent is assumed to be data parallel, so the local
        // size of the sample dimension should be equal to
        // the local width of previous activations. The check only
        // matters for split root processes as the rest just hold
        // invalid copy of the root data.
        if (get_original_prev_activations().is_split_root()) {
          assert_eq(
              (int)get_original_prev_activations().get_local_shape()[-1],
              l.get_prev_activations().LocalWidth());
        }
      }
    }
  }
  // TODO: Needs to check other output tensors
  if (child_copy_required(0) && get_original_activations().is_split_root()) {
    assert_eq((int)get_original_activations().get_local_shape()[-1],
              l.get_activations().LocalWidth());
  }
  ensure_prev_activations();
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::fp_postprocess() {
  copy_out_activations();
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::bp_setup(El::Int mini_batch_size) {
  const auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  set_error_signals_outermost_dimension(mini_batch_size);
  for (int i = 0; i < l.get_num_children(); ++i) {
    if (child_copy_required(i) || child_shuffle_required(i)) {
      auto &original_input = get_original_prev_error_signals(i);
      if (i != 0) {
        LBANN_ERROR("Copyout non-first tensor not supported");
      }
      if (child_copy_required(i) && original_input.is_split_root()) {
        assert_eq(
            (int)original_input.get_local_shape()[-1],
            l.get_prev_error_signals().LocalWidth());
      }
    }
    // TODO: Check other input tensors
    if (i == 0) {
      if (parent_copy_required(i) &&
          get_original_error_signals().is_split_root()) {
        assert_eq((int)get_original_error_signals().get_local_shape()[-1],
                  l.get_error_signals().LocalWidth());
      }
    }
  }
  ensure_prev_error_signals();
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::bp_postprocess() {
  copy_out_error_signals();
}

namespace {
template <typename TensorDataType>
dc::TensorShuffler<TensorDataType> &get_shuffler(
    const Layer &layer,
    std::array<dc::TensorShuffler<TensorDataType>*, 4> &shufflers,
    const dc::TensorDev<TensorDataType> &src,
    const dc::TensorDev<TensorDataType> &dst,
    const size_t max_mini_batch_size) {
  const auto& c = static_cast<sgd_execution_context&>(
      layer.get_model()->get_execution_context());
  const auto& mini_batch_size = c.get_current_mini_batch_size();
  int shuffler_idx = -1;
  if (max_mini_batch_size == mini_batch_size) {
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
  return get_shuffler(layer(), m_prev_activations_shufflers, src, dst,
                      get_max_mini_batch_size());
}

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_activations_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_activations_shufflers, src, dst,
                      get_max_mini_batch_size());
}

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_prev_error_signals_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_prev_error_signals_shufflers, src, dst,
                      get_max_mini_batch_size());
}

template <typename TensorDataType>
dc::TensorShuffler<TensorDataType>& data_type_distconv_adapter<TensorDataType>::
get_error_signals_shuffler(
    const dc::TensorDev<TensorDataType> &src, const dc::TensorDev<TensorDataType> &dst) {
  return get_shuffler(layer(), m_error_signals_shufflers, src, dst,
                      get_max_mini_batch_size());
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::ensure_prev_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (!(parent_copy_required(i) || parent_shuffle_required(i))) {
      continue;
    }
    if (i != 0) {
      LBANN_ERROR(layer().get_name(), ": copyin of non-first tensor not supported");
    }
    if (parent_copy_required(i)) {
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
        hydrogen::cuda::GetDefaultStream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::copy_out_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_children(); ++i) {
    if (!child_copy_required(i)) continue;
    if (i != 0) {
      LBANN_ERROR(layer().get_name(), ": Copyout of non-first tensor not supported");
    }
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
        hydrogen::cuda::GetDefaultStream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::ensure_prev_error_signals() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_children(); ++i) {
    if (!(child_copy_required(i) || child_shuffle_required(i))) {
      continue;
    }
    if (i != 0) {
      LBANN_ERROR(layer().get_name(), ": copyin of non-first tensor not supported");
    }
    if (child_copy_required(i)) {
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
        hydrogen::cuda::GetDefaultStream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::copy_out_error_signals() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (!parent_copy_required(i)) continue;
    if (i != 0) {
      LBANN_ERROR(layer().get_name(), ": Copyout of non-first tensor not supported");
    }
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
        hydrogen::cuda::GetDefaultStream());
  }
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_activations() const {
  dc::dump_tensor(get_activations(), get_name() + "_activations");
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_original_activations() {
  const auto &l = dynamic_cast<const data_type_layer<TensorDataType>&>(layer());
  assert0(dc::tensor::View(
      get_original_activations(), l.get_activations().LockedBuffer()));
  dc::dump_tensor(get_original_activations(),
                  get_name() + "_activations_original");
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_error_signals() const {
  dc::dump_tensor(get_error_signals(0), get_name() + "_error_signals");
}

template <typename TensorDataType>
void data_type_distconv_adapter<TensorDataType>::dump_original_error_signals() {
  const auto &l = dynamic_cast<const data_type_layer<TensorDataType>&>(layer());
  assert0(dc::tensor::View(
      get_original_activations(), l.get_activations().LockedBuffer()));
  dc::dump_tensor(get_original_error_signals(0),
                  get_name() +  "_error_signals_original");
}

template <typename TensorDataType>
size_t data_type_distconv_adapter<TensorDataType>::get_max_mini_batch_size() const {
  return layer().get_model()->get_max_mini_batch_size_distconv();
}

#define PROTO(T)                                \
  template class data_type_distconv_adapter<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
