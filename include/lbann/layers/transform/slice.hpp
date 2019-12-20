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
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  slice_layer(lbann_comm *comm,
              El::Int slice_dim,
              std::vector<El::Int> slice_points);
  slice_layer(const slice_layer& other);
  slice_layer& operator=(const slice_layer& other);

  slice_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

  /** Get slice points. */
  std::vector<El::Int>& get_slice_points() { return m_slice_points; }
  /** Get slice points (const). */
  std::vector<El::Int> get_slice_points() const { return m_slice_points; }

protected:

  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;

  void fp_setup_outputs(El::Int mini_batch_size) override;
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Tensor dimension to slice. */
  El::Int m_slice_dim;
  /** Slice points for each child layer. */
  std::vector<El::Int> m_slice_points;

  /** View into input tensor. */
  std::unique_ptr<AbsDistMatrixType> m_input_v;
  /** View into output tensor. */
  std::unique_ptr<AbsDistMatrixType> m_output_v;

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
  cuda::event_wrapper m_workspace_event;
#endif // LBANN_HAS_GPU

  template <typename U, El::Device D>
  friend void fp_setup_outputs_impl(slice_layer<U,Layout,D>&);
  template <typename U>
  friend void fp_compute_impl(slice_layer<U,Layout,Device>&);

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType,Layout,Device>::slice_layer(
  lbann_comm *comm,
  El::Int slice_dim,
  std::vector<El::Int> slice_points)
  : data_type_layer<TensorDataType>(comm),
  m_slice_dim(slice_dim),
  m_slice_points(slice_points) {
  this->m_expected_num_child_layers = -1; // No limit on children
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType,Layout,Device>::slice_layer(
  const slice_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_slice_dim(other.m_slice_dim),
    m_slice_points(other.m_slice_points) {
  m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
  m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType,Layout,Device>& slice_layer<TensorDataType,Layout,Device>::operator=(
  const slice_layer& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_slice_dim = other.m_slice_dim;
  m_slice_points = other.m_slice_points;
  m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
  m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
slice_layer<TensorDataType, Layout,Device>* slice_layer<TensorDataType,Layout,Device>::copy() const {
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
  std::stringstream ss;
  for (size_t i = 0; i < m_slice_points.size(); ++i) {
    ss << (i > 0 ? ", " : "") << m_slice_points[i];
  }
  desc.add("Slice points", ss.str());
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
  const auto& input = this->get_prev_activations();
  m_input_v.reset(input.Construct(input.Grid(), input.Root()));
  m_output_v.reset(input.Construct(input.Grid(), input.Root()));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();
  const auto& input_dims = this->get_input_dims();
  const auto& num_outputs = this->get_num_children();

  // Check that slice parameters are valid
  if (m_slice_dim < 0 || m_slice_dim >= (El::Int) input_dims.size()) {
    LBANN_ERROR(get_type()," layer \"",this->get_name(),"\" ",
                "has ",input_dims.size()," dimensions, ",
                "but attempted to slice along dimension ",m_slice_dim);
  }
  if ((int) m_slice_points.size() <= num_outputs) {
    LBANN_ERROR(get_type()," layer \"",this->get_name(),"\" ",
                "requires more slice points than output tensors ",
                "(found ",m_slice_points.size()," slice points ",
                "and ",this->m_child_layers.size()," output tensors)");
  }
  if (!std::is_sorted(m_slice_points.begin(), m_slice_points.end())) {
    LBANN_ERROR(get_type()," layer \"",this->get_name(),"\" ",
                "has unsorted slice points");
  }
  if (m_slice_points.front() < 0
      || m_slice_points.back() > input_dims[m_slice_dim]) {
    std::stringstream err;
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "expects slice points in the range "
        << "[0, " << input_dims[m_slice_dim] << "], "
        << "but found an invalid slice point ";
    if (m_slice_points.front() < 0) {
      err << "(" << m_slice_points.front() << ")";
    } else {
      err << "(" << m_slice_points.back() << ")";
    }
    LBANN_ERROR(err.str());
  }

  // Model-parallel implementation only supports flat data
  if (Layout == data_layout::MODEL_PARALLEL && input_dims.size() != 1) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "attempted to slice along dimension ",m_slice_dim,", ",
                "but model-parallel slice layer only supports flat data");
  }

  // Set output tensor dimensions
  auto output_dims = input_dims;
  for (int i = 0; i < num_outputs; ++i) {
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
  if (num_outputs == 1) {
    El::LockedView(l.get_activations(0), input);
  }
  else {
    for (size_t j=0; j<num_outputs; ++j) {
      auto& output = l.get_activations(j);
      output.AlignWith(input);
      El::Zeros(output, l.get_output_size(j), input.Width());
    }
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
  const auto& num_outputs = this->get_num_children();
  const auto& input_dims = this->get_input_dims();

  // Initialize gradient w.r.t. input tensor
  auto& gradient_wrt_input = this->get_error_signals();
  gradient_wrt_input.Empty(false);
  gradient_wrt_input.AlignWith(this->get_prev_activations());
  gradient_wrt_input.Resize(this->get_input_size(), mini_batch_size);
  if (m_slice_points[0] != 0
      || m_slice_points[num_outputs] != input_dims[m_slice_dim]) {
    El::Zero(gradient_wrt_input);
  }

  // Divide input tensor into unit slices along slice dimension
  // Note: Each unit slice is divided into contiguous "unit blocks"
  const auto& input_num_unit_slices = input_dims[m_slice_dim];
  const auto& blocks_per_slice
    = std::accumulate(&input_dims[0], &input_dims[m_slice_dim],
                      1, std::multiplies<int>());
  const auto& unit_block_size
    = std::accumulate(input_dims.begin() + m_slice_dim + 1,
                      input_dims.end(),
                      1, std::multiplies<int>());
  const auto& input_block_stride = (input_num_unit_slices
                                    * unit_block_size);

  // Populate slices of gradient w.r.t. input tensor
  for (int i = 0; i < num_outputs; ++i) {
    const auto& output_dims = this->get_output_dims(i);
    const auto& gradient_wrt_output = this->get_prev_error_signals(i);

    // Divide output tensor into unit slices
    const auto& output_num_unit_slices = output_dims[m_slice_dim];

    // Merge unit slices
    const auto& block_size = output_num_unit_slices * unit_block_size;
    const auto& input_block_offset = m_slice_points[i] * unit_block_size;

    // Populate gradient w.r.t. input tensor one block at a time
    for (int block = 0; block < blocks_per_slice; ++block) {
      const auto& input_offset = (input_block_offset
                                  + block * input_block_stride);
      const auto& output_offset = block * block_size;
      El::LockedView(*m_output_v, gradient_wrt_output,
                     El::IR(output_offset, output_offset + block_size),
                     El::ALL);
      El::View(*m_input_v, gradient_wrt_input,
               El::IR(input_offset, input_offset + block_size),
               El::ALL);
      El::Copy(*m_output_v, *m_input_v);
    }

  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType,Layout,Device>::bp_compute() {
}

#ifndef LBANN_SLICE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)             \
  extern template class slice_layer<        \
    T, data_layout::DATA_PARALLEL, Device>; \
  extern template class slice_layer<        \
    T, data_layout::MODEL_PARALLEL, Device>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_SLICE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_SLICE_HPP_INCLUDED
