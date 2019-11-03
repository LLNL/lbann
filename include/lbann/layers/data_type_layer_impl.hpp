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

namespace lbann {

template <typename TensorDataType>
data_type_layer<TensorDataType>::data_type_layer(const data_type_layer<TensorDataType>& other) :
  Layer(other),
  m_weights(other.m_weights),
  m_frozen(other.m_frozen),
  m_output_dims_list(other.m_output_dims_list),
  m_hint_layer(other.m_hint_layer) {

  // Deep matrix copies
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }

}

template <typename TensorDataType>
data_type_layer<TensorDataType>& data_type_layer<TensorDataType>::operator=(const data_type_layer<TensorDataType>& other) {
  Layer::operator=(other);

  // Shallow copies
  m_weights = other.m_weights;
  m_frozen = other.m_frozen;
  m_output_dims_list = other.m_output_dims_list;
  m_hint_layer = other.m_hint_layer;

  // Deep matrix copies
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }

  return *this;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::replace_weights(data_type_layer<TensorDataType>* other_layer) {
  if (other_layer == nullptr) {
    LBANN_ERROR("attempted to add null pointer as a replacement layer");
  }

  const std::vector<data_type_weights<TensorDataType> *> other_layer_weights = other_layer->get_weights();
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_weights[i]->set_values(other_layer_weights[i]->get_values());
  }

}

// ===================================================================
// Tensor dimension access functions
// ===================================================================

template <typename TensorDataType>
std::vector<int> data_type_layer<TensorDataType>::get_input_dims(int input_index) const {

  // Get parent layer
  const auto& num_inputs = get_num_parents();
  if (input_index < 0 || input_index >= num_inputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid input tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << input_index << ", but there are "
        << num_inputs << " input tensors)";
    LBANN_ERROR(err.str());
  } else if (m_parent_layers[input_index] == nullptr) {
    std::stringstream err;
    err << "layer \"" << get_name() << "\" "
        << "has a null pointer to parent layer "
        << "(index " << input_index << ")";
    LBANN_ERROR(err.str());
  }
  const auto& parent = *m_parent_layers[input_index];

  // Get dimensions of corresponding output tensor in parent layer
  const auto num_parent_outputs = parent.get_num_children();
  // const int parent_output_index = (std::find(parent.m_child_layers.begin(),
  //                                            parent.m_child_layers.end(),
  //                                            this)
  //                                  - parent.m_child_layers.begin());
  const int parent_output_index = parent.find_layer_index(this);
  if (parent_output_index >= num_parent_outputs) {
    std::stringstream err;
    err << "layer \"" << parent.get_name() << "\" is a parent of "
        << "layer \"" << get_name() << "\", but "
        << "\"" << get_name() << "\" is not a child of "
        << "\"" << parent.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return dynamic_cast<const data_type_layer<TensorDataType>&>(parent).get_output_dims(parent_output_index);

}

template <typename TensorDataType>
int data_type_layer<TensorDataType>::get_input_size(int input_index) const {
  const auto& dims = get_input_dims(input_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
  }
}

template <typename TensorDataType>
std::vector<int> data_type_layer<TensorDataType>::get_output_dims(int output_index) const {
  const auto num_outputs = get_num_children();
  if ((int) m_output_dims_list.size() != num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of output tensor "
        << "in layer \"" << get_name() << "\" "
        << "before they are initialized";
    LBANN_ERROR(err.str());
  } else if (output_index < 0 || output_index >= num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid output tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << output_index << ", but there are "
        << num_outputs << " output tensors)";
    LBANN_ERROR(err.str());
  }
  return m_output_dims_list[output_index];
}

template <typename TensorDataType>
int data_type_layer<TensorDataType>::get_output_size(int output_index) const {
  const auto& dims = get_output_dims(output_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::set_output_dims(std::vector<int> dims, int output_index) {
  if ((int) m_output_dims_list.size() != get_num_children()
      || (int) m_output_dims_list.size() <= output_index) {
    // Handles case where dims are set before child layers are set
    m_output_dims_list.resize(std::max(get_num_children(),
                                       output_index + 1));
  }
  m_output_dims_list[output_index] = dims;
}

// ===================================================================
// Tensor access functions
// ===================================================================

// Accessing distributed matrices
template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_prev_activations(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_inputs.size() << " previous activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_inputs[parent_index];
}

template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_activations(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_outputs.size() << " activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_outputs[child_index];
}

template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_prev_error_signals(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_gradient_wrt_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_gradient_wrt_outputs.size() << " previous error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_gradient_wrt_outputs[child_index];
}

template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_error_signals(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_gradient_wrt_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_gradient_wrt_inputs.size() << " error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_gradient_wrt_inputs[parent_index];
}

// Accessing non-const distributed matrices
// Note: Using idiom from Item 3, p. 23 in "Effective C++", 3rd ed.,
// by Scott Meyers.
template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_activations(int child_index) {
  return const_cast<El::AbstractDistMatrix<TensorDataType>&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_activations(child_index));
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_error_signals(int parent_index) {
  return const_cast<El::AbstractDistMatrix<TensorDataType>&>(static_cast<const data_type_layer<TensorDataType>&>(*this).get_error_signals(parent_index));
}

// Accessing local matrices
template <typename TensorDataType>
El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_activations(int child_index) {
  return get_activations(child_index).Matrix();
}
template <typename TensorDataType>
El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) {
  return get_error_signals(parent_index).Matrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_prev_activations(int parent_index) const {
  return get_prev_activations(parent_index).LockedMatrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_activations(int child_index) const {
  return get_activations(child_index).LockedMatrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_prev_error_signals(int child_index) const {
  return get_prev_error_signals(child_index).LockedMatrix();
}
template <typename TensorDataType>
const El::AbstractMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_local_error_signals(int parent_index) const {
  return get_error_signals(parent_index).LockedMatrix();
}

// Accessing matrices corresponding to parent/child layer
template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_activations(const data_type_layer<TensorDataType>& child) const {
  const int child_index = (std::find(m_child_layers.begin(),
                                     m_child_layers.end(),
                                     &child)
                           - m_child_layers.begin());
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
template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_layer<TensorDataType>::get_error_signals(const data_type_layer<TensorDataType>& parent) const {
  const int parent_index = (std::find(m_parent_layers.begin(),
                                      m_parent_layers.end(),
                                      &parent)
                           - m_parent_layers.begin());
  if (parent_index >= get_num_parents()) {
    std::stringstream err;
    err << "attempted to get error signal tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << parent.get_name() << "\", "
        << "which is not a parent layer";
    LBANN_ERROR(err.str());
  }
  return get_error_signals(parent_index);
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::freeze() {
  m_frozen = true;
  for(auto& w : m_weights) {
    w->freeze();
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::unfreeze() {
  m_frozen = false;
  for(auto& w : m_weights) {
    w->unfreeze();
  }
}

template <typename TensorDataType>
bool data_type_layer<TensorDataType>::is_frozen() const {
  for(auto& w : m_weights) {
    if (w->is_frozen() != m_frozen) {
      LBANN_ERROR("layer and weights of them are inconsistently frozen");
    }
  }
  return m_frozen;
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_dims() {
  m_output_dims_list.resize(get_num_children());
  if (m_hint_layer != nullptr) {
    const auto& hint_dims = m_hint_layer->get_output_dims();
    for (auto& output_dims : m_output_dims_list) {
      output_dims = hint_dims;
    }
  } else if (get_num_parents() > 0) {
    const auto& input_dims = get_input_dims();
    for (auto& output_dims : m_output_dims_list) {
      if (output_dims.empty()) {
        output_dims = input_dims;
      }
    }
  }
}

template <typename TensorDataType>
void data_type_layer<TensorDataType>::setup_matrices(const El::Grid& grid) {

  // Destroy previously setup matrices
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();

  // Construct matrices
  m_inputs.resize(get_num_parents());
  m_outputs.resize(get_num_children());
  m_gradient_wrt_outputs.resize(get_num_children());
  m_gradient_wrt_inputs.resize(get_num_parents());
  for (int i = 0; i < get_num_parents(); ++i) {
    m_inputs[i] = construct_matrix(grid, "input", i);
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_outputs[i] = construct_matrix(grid, "output", i);
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_gradient_wrt_outputs[i]
      = construct_matrix(grid, "gradient_wrt_output", i);
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    m_gradient_wrt_inputs[i]
      = construct_matrix(grid, "gradient_wrt_input", i);
  }
}

template <typename TensorDataType>
std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> data_type_layer<TensorDataType>::construct_matrix(const El::Grid& grid,
                                                                              std::string type,
                                                                              El::Int index) {

  // Choose matrix distribution
  El::Distribution col_dist, row_dist;
  El::DistWrap wrap;
  El::Device device = this->get_device_allocation();
  switch (get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    col_dist = El::STAR;
    row_dist = El::VC;
    wrap     = El::ELEMENT;
    break;
  case data_layout::MODEL_PARALLEL:
    col_dist = El::MC;
    row_dist = El::MR;
    wrap     = El::ELEMENT;
    break;
  default: LBANN_ERROR("invalid data layout");
  }

  // Construct matrix
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> mat;
  mat.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(grid, 0,
                                    col_dist, row_dist, wrap, device));

#ifdef LBANN_HAS_GPU
  // Allocate GPU memory with the CUDA API
  if (device == El::Device::GPU) { mat->Matrix().SetMemoryMode(0); }
  // Use pinned memory for data on the host.
  if (device == El::Device::CPU) { mat->Matrix().SetMemoryMode(1); }
#endif // LBANN_HAS_GPU

  return mat;
}

}
