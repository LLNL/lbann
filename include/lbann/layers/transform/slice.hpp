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

#ifndef LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"

namespace lbann {

/** @brief Slice tensor along a specified dimension.
 *
 *  Suppose we slice a @f$ D_1 \times\cdots\times D_n @f$ input tensor
 *  along the dimension @f$ k @f$. We specify slice points
 *  @f$ s_1,\cdots,s_\ell @f$, which are strictly increasing and have
 *  @f$ s_1 = 0 @f$ and @f$ s_\ell=D_k @f$. The @f$ i @f$th output
 *  tensor is then a
 *  @f$ D_1 \times\cdots
 *    \times D_{i-1}\times (s_i - s_{i-1}) \times D_{i+1} \times
 *    \cdots\times D_n @f$
 *  tensor.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class slice_layer : public data_type_layer<TensorDataType> {
public:

  slice_layer(lbann_comm *comm);
  slice_layer(const slice_layer& other) = default;
  slice_layer& operator=(const slice_layer& other) = default;

  slice_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

  void setup_slice_points(size_t slice_dim,
                          std::vector<size_t> slice_points) {
    m_slice_dim = slice_dim;
    m_slice_points = std::move(slice_points);
  }

  void setup_slice_points(size_t slice_dim,
                          bool set_slice_points_from_data_reader,
                          const slice_points_mode var_category) {
    m_slice_dim = slice_dim;
    m_set_slice_points_from_data_reader = set_slice_points_from_data_reader;
    m_var_category = var_category;
  }

protected:

  friend class cereal::access;
  slice_layer()
    : slice_layer(nullptr)
  {}

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_setup_outputs(El::Int mini_batch_size) override;
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Tensor dimension to slice. */
  size_t m_slice_dim;
  /** Slice points for each child layer. */
  std::vector<size_t> m_slice_points;
  /** Slice points are automatically defined by the data reader */
  bool m_set_slice_points_from_data_reader;
  /** Category for retrieving slice points from data reader */
  slice_points_mode m_var_category;

#ifdef LBANN_HAS_GPU
  /** @brief Workspace buffer.
   *
   *  Parameters for CUDA kernels are copied into this buffer and
   *  asynchronously transferred to GPU.
   */
  std::vector<unsigned char> m_workspace;
  /** @brief CUDA event for workspace buffer.
   *
   *  Makes sure asynchronous GPU memory transfers are completed
   *  before modifying workspace buffer.
   */
  gpu_lib::event_wrapper m_workspace_event;
#endif // LBANN_HAS_GPU

  template <typename U, El::Device D>
  friend void fp_setup_outputs_impl(slice_layer<U,Layout,D>&);
  template <typename U>
  friend void fp_compute_impl(slice_layer<U,Layout,Device>&);
  template <typename U>
  friend void bp_compute_impl(slice_layer<U,Layout,Device>&);

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType,Layout,Device>::slice_layer(lbann_comm *comm)
  : data_type_layer<TensorDataType>(comm),
  m_set_slice_points_from_data_reader(false),
  m_var_category(slice_points_mode::NA) {
  this->m_expected_num_child_layers = -1; // No limit on children
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType,Layout,Device>* slice_layer<TensorDataType,Layout,Device>::copy() const {
  return new slice_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string slice_layer<TensorDataType,Layout,Device>::get_type() const {
  return "slice";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout slice_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device slice_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description slice_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Slice dimension", m_slice_dim);
  std::ostringstream ss;
  for (size_t i = 0; i < m_slice_points.size(); ++i) {
    ss << (i > 0 ? ", " : "") << m_slice_points[i];
  }
  desc.add("Slice points", ss.str());
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Setup the slice points if they are to be established by the data reader
  if(m_set_slice_points_from_data_reader) {
    std::vector<size_t> slice_points;
    std::string slice_point_method_name = "'get_slice_points_from_reader'";
    for (auto& slice_point
           : dr_metadata.slice_points[m_var_category]) {
      slice_points.push_back(slice_point);
    }

    if (slice_points.size() < 2u) {
      LBANN_ERROR(slice_point_method_name, " is not supported by the reader.");
      return;
    }
    m_slice_points = std::move(slice_points);
  }

  // Check that slice parameters are valid
  const auto& input_dims = this->get_input_dims();
  const size_t num_outputs = this->get_num_children();
  if (m_slice_dim >= input_dims.size()) {
    std::ostringstream err;
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "is slicing along dimension " << m_slice_dim << ", "
        << "but it has a " << input_dims.size() << "-D input tensor "
        << "(parent layer \"" << this->get_parent_layers()[0]->get_name() << "\" "
        << "outputs with dimensions ";
    for (size_t d=0; d<input_dims.size(); ++d) {
      err << (d>0 ? " x " : "") << input_dims[d];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
  if (m_slice_points.size() <= num_outputs) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "has ",num_outputs," children, "
                "but only ",m_slice_points.size()," slice points");
  }
  if (!std::is_sorted(m_slice_points.begin(), m_slice_points.end())) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "has unsorted slice points");
  }
  if (m_slice_points.back() > static_cast<size_t>(input_dims[m_slice_dim])) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "has a slice point of ",m_slice_points.back(),", ",
                "which is outside the expected range "
                "[0 ",input_dims[m_slice_dim],"]");
  }

  // Model-parallel implementation only supports flat data
  if (Layout == data_layout::MODEL_PARALLEL && input_dims.size() != 1) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "attempted to slice along dimension ",m_slice_dim,", ",
                "but model-parallel slice layer only supports flat data");
  }

  // Set output tensor dimensions
  auto output_dims = input_dims;
  for (size_t i = 0; i < num_outputs; ++i) {
    output_dims[m_slice_dim] = m_slice_points[i+1] - m_slice_points[i];
    this->set_output_dims(output_dims, i);
  }

}

template <typename TensorDataType, El::Device Device>
void fp_setup_outputs_impl(
  slice_layer<TensorDataType,data_layout::MODEL_PARALLEL,Device>& l) {

  // Slice Elemental matrices
  // Note: Assume each mini-batch sample is flat.
  const size_t num_outputs = l.get_num_children();
  const auto& input = l.get_prev_activations();
  size_t offset = l.m_slice_points.front();
  for (size_t j=0; j<num_outputs; ++j) {
    auto& output = l.get_activations(j);
    const auto& output_size = l.get_output_size(j);
    El::LockedView(output, input,
                   El::IR(offset, offset+output_size), El::ALL);
    offset += output_size;
  }

}

template <typename TensorDataType, El::Device Device>
void fp_setup_outputs_impl(
  slice_layer<TensorDataType,data_layout::DATA_PARALLEL,Device>& l) {

  const size_t num_outputs = l.get_num_children();
  const auto& input = l.get_prev_activations();
  for (size_t j=0; j<num_outputs; ++j) {
    auto& output = l.get_activations(j);
    output.AlignWith(input);
    output.Resize(l.get_output_size(j), input.Width());
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::fp_setup_outputs(El::Int mini_batch_size) {
  fp_setup_outputs_impl(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::fp_compute() {
  fp_compute_impl(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
  const auto& output0_grad = this->get_prev_error_signals(0);
  auto& input_grad = this->get_error_signals();
  input_grad.Empty(false);
  input_grad.AlignWith(output0_grad);
  El::Zeros(input_grad, this->get_input_size(), output0_grad.Width());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::bp_compute() {
  bp_compute_impl(*this);
}

#ifndef LBANN_SLICE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)             \
  extern template class slice_layer<        \
    T, data_layout::DATA_PARALLEL, Device>; \
  extern template class slice_layer<        \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SLICE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED
