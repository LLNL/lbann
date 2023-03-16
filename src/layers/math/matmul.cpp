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

#define LBANN_MATMUL_LAYER_INSTANTIATE
#include "lbann/layers/math/matmul.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_GPU
#include "lbann/utils/gpu/helpers.hpp"
#endif // LBANN_HAS_GPU
#include "lbann/proto/layers.pb.h"
#include <iostream>

namespace lbann {

#ifdef LBANN_HAS_DISTCONV

// =========================================================
// DistConv-Adapter member functions
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_distconv_adapter<TensorDataType, Layout, Device>::
  setup_distributions(tensor_overlap_constraints& constraints)
{

  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);

  for (auto& d : this->m_prev_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto& d : this->m_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto& d : this->m_prev_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto& d : this->m_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
  size_t workspace_capacity)
{
  data_type_distconv_adapter<TensorDataType>::setup_layer(workspace_capacity);

  m_matmul_operator =
    std::make_unique<dc::MatMul<TensorDataType>>(dc::get_backend());
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_distconv_adapter<TensorDataType, Layout, Device>::fp_compute()
{
  auto& layer =
    dynamic_cast<matmul_layer<TensorDataType, Layout, Device>&>(this->layer());
  m_matmul_operator->forward(this->get_prev_activations(0),
                             this->get_prev_activations(1),
                             this->get_activations(0),
                             layer.m_transpose_a,
                             layer.m_transpose_b);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_distconv_adapter<TensorDataType, Layout, Device>::bp_compute()
{
  auto& layer =
    dynamic_cast<matmul_layer<TensorDataType, Layout, Device>&>(this->layer());
  m_matmul_operator->backward(this->get_prev_activations(0),
                              this->get_prev_activations(1),
                              this->get_prev_error_signals(),
                              this->get_error_signals(0),
                              this->get_error_signals(1),
                              layer.m_transpose_a,
                              layer.m_transpose_b);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape matmul_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  const auto& layer =
    dynamic_cast<const matmul_layer<TensorDataType, Layout, Device>&>(
      this->layer());
  const auto output_shape =
    ::distconv::get_matmul_local_tensor_shape(this->get_prev_activations(0),
                                              this->get_prev_activations(1),
                                              layer.m_transpose_a,
                                              layer.m_transpose_b);
  return output_shape;
}
// =============================================================
// DistConv-enabled MatMul member functions
// =============================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool matmul_layer<TensorDataType, Layout, Device>::is_distconv_supported() const
{
  return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::setup_distconv_adapter(
  const DataReaderMetaData& dr_metadata)
{
  this->get_distconv_adapter_ptr() =
    std::make_unique<matmul_distconv_adapter<TensorDataType, Layout, Device>>(
      *this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
const matmul_distconv_adapter<TensorDataType, Layout, Device>&
matmul_layer<TensorDataType, Layout, Device>::get_distconv_adapter() const
{
  return dynamic_cast<
    const matmul_distconv_adapter<TensorDataType, Layout, Device>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
matmul_distconv_adapter<TensorDataType, Layout, Device>&
matmul_layer<TensorDataType, Layout, Device>::get_distconv_adapter()
{
  return const_cast<matmul_distconv_adapter<TensorDataType, Layout, Device>&>(
    static_cast<const matmul_layer<TensorDataType, Layout, Device>&>(*this)
      .get_distconv_adapter());
}

#endif //  LBANN_HAS_DISTCONV

template <typename TensorDataType>
void fp_compute_impl(
  matmul_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l,
  bool transpose_input0,
  bool transpose_input1)
{

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input0 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& local_output = dynamic_cast<LocalMat&>(l.get_local_activations());
  const auto& local_mini_batch_size = local_input0.Width();

  if (!local_input0.Contiguous() || !local_input1.Contiguous() ||
      !local_output.Contiguous()) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "has non-contiguous data buffers");
  }
  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();
  // Check if matrix is 3D or 2D
  const El::Int mat_depth =
    (input0_dims.size() > 2) ? *(input0_dims.rbegin() + 2) : 1;

  const El::Int input0_height = *(input0_dims.rbegin() + 1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_height = *(input1_dims.rbegin() + 1);
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin() + 1);
  const El::Int output_width = *(output_dims.rbegin());

  // Compute matrix multiplication for each mini-batch sample
  // Note: Elemental matrices are in Fortran layout while LBANN
  // tensors are in C layout.

  // Calculate stride for each matrix depending on the depth
  const auto input0_stride = input0_height * input0_width;
  const auto input1_stride = input1_height * input1_width;
  const auto output_stride = output_height * output_width;

  LBANN_OMP_PARALLEL_FOR
  for (El::Int j = 0; j < mat_depth; ++j) {
    auto input0_buffer_start = j * input0_stride;
    auto input1_buffer_start = j * input1_stride;
    auto output_buffer_start = j * output_stride;
    for (El::Int i = 0; i < local_mini_batch_size; ++i) {
      LocalMat input0_v, input1_v, output_v;
      input0_v.LockedAttach(input0_width,
                            input0_height,
                            local_input0.LockedBuffer(input0_buffer_start, i),
                            input0_width);
      input1_v.LockedAttach(input1_width,
                            input1_height,
                            local_input1.LockedBuffer(input1_buffer_start, i),
                            input1_width);
      output_v.Attach(output_width,
                      output_height,
                      local_output.Buffer(output_buffer_start, i),
                      output_width);
      El::Gemm(transpose_input1 ? El::TRANSPOSE : El::NORMAL,
               transpose_input0 ? El::TRANSPOSE : El::NORMAL,
               El::TypeTraits<TensorDataType>::One(),
               input1_v,
               input0_v,
               El::TypeTraits<TensorDataType>::Zero(),
               output_v);
    }
  }
}

template <typename TensorDataType>
void bp_compute_impl(
  matmul_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l,
  bool transpose_input0,
  bool transpose_input1)
{

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input0 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  const auto& local_output_grad =
    dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& local_input0_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& local_input1_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(1));
  const auto& local_mini_batch_size = local_input0.Width();
  if (!local_input0.Contiguous() || !local_input1.Contiguous() ||
      !local_output_grad.Contiguous() || !local_input0_grad.Contiguous() ||
      !local_input1_grad.Contiguous()) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "has non-contiguous data buffers");
  }
  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();
  // Check if matrix is 3D or 2D
  const El::Int mat_depth =
    (input0_dims.size() > 2) ? *(input0_dims.rbegin() + 2) : 1;

  const El::Int input0_height = *(input0_dims.rbegin() + 1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_height = *(input1_dims.rbegin() + 1);
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin() + 1);
  const El::Int output_width = *(output_dims.rbegin());

  // Compute gradients for each mini-batch sample
  // Note: Elemental matrices are in Fortran layout while LBANN
  // tensors are in C layout.

  // Calculate stride for each matrix depending on the depth
  const auto input0_stride = input0_height * input0_width;
  const auto input1_stride = input1_height * input1_width;
  const auto output_stride = output_height * output_width;

  LBANN_OMP_PARALLEL_FOR
  for (El::Int j = 0; j < mat_depth; ++j) {
    auto input0_buffer_start = j * input0_stride;
    auto input1_buffer_start = j * input1_stride;
    auto output_buffer_start = j * output_stride;
    for (El::Int i = 0; i < local_mini_batch_size; ++i) {
      LocalMat input0_v, input1_v, output_grad_v, input0_grad_v, input1_grad_v;
      input0_v.LockedAttach(input0_width,
                            input0_height,
                            local_input0.LockedBuffer(input0_buffer_start, i),
                            input0_width);
      input1_v.LockedAttach(input1_width,
                            input1_height,
                            local_input1.LockedBuffer(input1_buffer_start, i),
                            input1_width);
      output_grad_v.LockedAttach(
        output_width,
        output_height,
        local_output_grad.LockedBuffer(output_buffer_start, i),
        output_width);
      input0_grad_v.Attach(input0_width,
                           input0_height,
                           local_input0_grad.Buffer(input0_buffer_start, i),
                           input0_width);
      input1_grad_v.Attach(input1_width,
                           input1_height,
                           local_input1_grad.Buffer(input1_buffer_start, i),
                           input1_width);
      if (transpose_input0) {
        El::Gemm(El::TRANSPOSE,
                 transpose_input1 ? El::TRANSPOSE : El::NORMAL,
                 El::TypeTraits<TensorDataType>::One(),
                 output_grad_v,
                 input1_v,
                 El::TypeTraits<TensorDataType>::Zero(),
                 input0_grad_v);
      }
      else {
        El::Gemm(transpose_input1 ? El::NORMAL : El::TRANSPOSE,
                 El::NORMAL,
                 El::TypeTraits<TensorDataType>::One(),
                 input1_v,
                 output_grad_v,
                 El::TypeTraits<TensorDataType>::Zero(),
                 input0_grad_v);
      }
      if (transpose_input1) {
        El::Gemm(transpose_input0 ? El::TRANSPOSE : El::NORMAL,
                 El::TRANSPOSE,
                 El::TypeTraits<TensorDataType>::One(),
                 input0_v,
                 output_grad_v,
                 El::TypeTraits<TensorDataType>::Zero(),
                 input1_grad_v);
      }
      else {
        El::Gemm(El::NORMAL,
                 transpose_input0 ? El::NORMAL : El::TRANSPOSE,
                 El::TypeTraits<TensorDataType>::One(),
                 output_grad_v,
                 input0_v,
                 El::TypeTraits<TensorDataType>::Zero(),
                 input1_grad_v);
      }
    }
  }
}

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void fp_compute_impl(
  matmul_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l,
  bool transpose_input0,
  bool transpose_input1)
{

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input0 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& local_output = dynamic_cast<LocalMat&>(l.get_local_activations());
  const auto& local_mini_batch_size = local_input0.Width();

  if (!local_input0.Contiguous() || !local_input1.Contiguous() ||
      !local_output.Contiguous()) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "has non-contiguous data buffers");
  }
  // Return immediately if nothing needs to be done
  if (local_mini_batch_size < 1) {
    return;
  }

  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();
  // Check if matrix is 3D or 2D
  const El::Int mat_depth =
    (input0_dims.size() > 2) ? *(input0_dims.rbegin() + 2) : 1;

  const El::Int input0_height = *(input0_dims.rbegin() + 1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_height = *(input1_dims.rbegin() + 1);
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin() + 1);
  const El::Int output_width = *(output_dims.rbegin());

  const auto num_matrices = mat_depth * local_mini_batch_size;
  const auto input0_stride = input0_height * input0_width;
  const auto input1_stride = input1_height * input1_width;
  const auto output_stride = output_height * output_width;

  // Compute gradients for each mini-batch sample
  // Note: cuBLAS expects matrices in Fortran layout while LBANN
  // tensors are in C layout.
  {
    using namespace hydrogen;
    auto multisync = MakeMultiSync(gpu::get_sync_info(local_output),
                                   gpu::get_sync_info(local_input0),
                                   gpu::get_sync_info(local_input1));

    gpu_blas::GemmStridedBatched(
      transpose_input1 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
      transpose_input0 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
      output_width,
      output_height,
      transpose_input0 ? input0_height : input0_width,
      El::TypeTraits<TensorDataType>::One(),
      local_input1.LockedBuffer(),
      input1_width,
      input1_stride,
      local_input0.LockedBuffer(),
      input0_width,
      input0_stride,
      El::TypeTraits<TensorDataType>::Zero(),
      local_output.Buffer(),
      output_width,
      output_stride,
      num_matrices,
      multisync);
  }
}
#endif // LBANN_HAS_GPU

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void bp_compute_impl(
  matmul_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l,
  bool transpose_input0,
  bool transpose_input1)
{

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input0 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 =
    dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  const auto& local_output_grad =
    dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& local_input0_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& local_input1_grad =
    dynamic_cast<LocalMat&>(l.get_local_error_signals(1));
  const auto& local_mini_batch_size = local_input0.Width();

  // Return immediately if nothing needs to be done
  if (local_mini_batch_size < 1) {
    return;
  }

  if (!local_input0.Contiguous() || !local_input1.Contiguous() ||
      !local_output_grad.Contiguous() || !local_input0_grad.Contiguous() ||
      !local_input1_grad.Contiguous()) {
    LBANN_ERROR(l.get_type(),
                " layer \"",
                l.get_name(),
                "\" ",
                "has non-contiguous data buffers");
  }
  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();

  // Check if matrix is 3D or 2D
  const El::Int mat_depth =
    (input0_dims.size() > 2) ? *(input0_dims.rbegin() + 2) : 1;

  const El::Int input0_height = *(input0_dims.rbegin() + 1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_height = *(input1_dims.rbegin() + 1);
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin() + 1);
  const El::Int output_width = *(output_dims.rbegin());

  const auto num_matrices = mat_depth * local_mini_batch_size;
  const auto input0_stride = input0_height * input0_width;
  const auto input1_stride = input1_height * input1_width;
  const auto output_stride = output_height * output_width;

  const auto input0_grad_stride = input0_stride;
  const auto input1_grad_stride = input1_stride;
  const auto output_grad_stride = output_stride;

  // Compute gradients for each mini-batch sample
  // Note: cuBLAS expects matrices in Fortran layout while LBANN
  // tensors are in C layout.
  {
    using namespace hydrogen;
    // The SyncInfo of the C matrix leads the way!
    auto multisync = MakeMultiSync(gpu::get_sync_info(local_input0_grad),
                                   gpu::get_sync_info(local_input1),
                                   gpu::get_sync_info(local_output_grad));

    if (transpose_input0) {
      gpu_blas::GemmStridedBatched(TransposeMode::TRANSPOSE,
                                   transpose_input1 ? TransposeMode::TRANSPOSE
                                                    : TransposeMode::NORMAL,
                                   input0_width,
                                   input0_height,
                                   output_width,
                                   El::TypeTraits<TensorDataType>::One(),
                                   local_output_grad.LockedBuffer(),
                                   output_width,
                                   output_grad_stride,
                                   local_input1.LockedBuffer(),
                                   input1_width,
                                   input1_stride,
                                   El::TypeTraits<TensorDataType>::Zero(),
                                   local_input0_grad.Buffer(),
                                   input0_width,
                                   input0_grad_stride,
                                   num_matrices,
                                   multisync);
    }
    else {
      gpu_blas::GemmStridedBatched(transpose_input1 ? TransposeMode::NORMAL
                                                    : TransposeMode::TRANSPOSE,
                                   TransposeMode::NORMAL,
                                   input0_width,
                                   input0_height,
                                   output_width,
                                   El::TypeTraits<TensorDataType>::One(),
                                   local_input1.LockedBuffer(),
                                   input1_width,
                                   input1_stride,
                                   local_output_grad.LockedBuffer(),
                                   output_width,
                                   output_grad_stride,
                                   El::TypeTraits<TensorDataType>::Zero(),
                                   local_input0_grad.Buffer(),
                                   input0_width,
                                   input0_grad_stride,
                                   num_matrices,
                                   multisync);
    }
  }
  {
    using namespace hydrogen;
    // The SyncInfo of the C matrix leads the way!
    auto multisync = MakeMultiSync(gpu::get_sync_info(local_input1_grad),
                                   gpu::get_sync_info(local_input0),
                                   gpu::get_sync_info(local_output_grad));
    if (transpose_input1) {
      gpu_blas::GemmStridedBatched(transpose_input0 ? TransposeMode::TRANSPOSE
                                                    : TransposeMode::NORMAL,
                                   TransposeMode::TRANSPOSE,
                                   input1_width,
                                   input1_height,
                                   output_height,
                                   El::TypeTraits<TensorDataType>::One(),
                                   local_input0.LockedBuffer(),
                                   input0_width,
                                   input0_stride,
                                   local_output_grad.LockedBuffer(),
                                   output_width,
                                   output_grad_stride,
                                   El::TypeTraits<TensorDataType>::Zero(),
                                   local_input1_grad.Buffer(),
                                   input1_width,
                                   input1_grad_stride,
                                   num_matrices,
                                   multisync);
    }
    else {
      gpu_blas::GemmStridedBatched(TransposeMode::NORMAL,
                                   transpose_input0 ? TransposeMode::NORMAL
                                                    : TransposeMode::TRANSPOSE,
                                   input1_width,
                                   input1_height,
                                   output_height,
                                   El::TypeTraits<TensorDataType>::One(),
                                   local_output_grad.LockedBuffer(),
                                   output_width,
                                   output_grad_stride,
                                   local_input0.LockedBuffer(),
                                   input0_width,
                                   input0_stride,
                                   El::TypeTraits<TensorDataType>::Zero(),
                                   local_input1_grad.Buffer(),
                                   input1_width,
                                   input1_grad_stride,
                                   num_matrices,
                                   multisync);
    }
  }
}
#endif // LBANN_HAS_GPU

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Input dimensions
  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);

  // Lambdas to help print error messages
  auto print_name = [this]() -> std::string {
    return this->get_type() + " layer \"" + this->get_name() + "\"";
  };
  auto print_inputs = [this, &input0_dims, &input1_dims]() -> std::string {
    auto print_dims = [](const decltype(input0_dims)& dims) -> std::string {
      std::ostringstream ss;
      for (size_t i = 0; i < dims.size(); ++i) {
        ss << (i > 0 ? "x" : "") << dims[i];
      }
      return ss.str();
    };
    const auto& parents = this->get_parent_layers();
    return lbann::build_string(parents[0]->get_type(),
                               " layer \"",
                               parents[0]->get_name(),
                               "\" ",
                               "outputs ",
                               print_dims(input0_dims),
                               ", ",
                               parents[1]->get_type(),
                               " layer \"",
                               parents[1]->get_name(),
                               "\" ",
                               "outputs ",
                               print_dims(input1_dims));
  };

  // Check input dimensions
  if (input0_dims.size() != input1_dims.size()) {
    LBANN_ERROR("input tensors in ",
                print_name(),
                " "
                "have different numbers of dimensions ",
                "(",
                print_inputs(),
                ")");
  }

  if (input0_dims.size() != 2 && input0_dims.size() != 3) {
    LBANN_ERROR("input tensors in ",
                print_name(),
                " are not 2D or 3D",
                "(",
                print_inputs(),
                ")");
  }
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    if (input0_dims.size() != 3) {
      LBANN_ERROR("input tensors in ",
                  print_name(),
                  " must be 3D when distconv is enabled",
                  "(",
                  print_inputs(),
                  ")");
    }
  }
#endif
  // Get matrix dimensions
  const auto input0_height = *(input0_dims.rbegin() + 1);
  const auto input0_width = *(input0_dims.rbegin());
  const auto input1_height = *(input1_dims.rbegin() + 1);
  const auto input1_width = *(input1_dims.rbegin());
  if ((m_transpose_a ? input0_height : input0_width) !=
      (m_transpose_b ? input1_width : input1_height)) {
    LBANN_ERROR("input tensors in ",
                print_name(),
                " ",
                "are not compatible with ",
                (m_transpose_a ? "T" : "N"),
                (m_transpose_b ? "T" : "N"),
                " matrix multiplication ",
                "(",
                print_inputs(),
                ")");
  }

  // Set output dimensions
  std::vector<int> output_dims(input0_dims);
  *(output_dims.rbegin() + 1) = (m_transpose_a ? input0_width : input0_height);
  *(output_dims.rbegin()) = (m_transpose_b ? input1_height : input1_width);
  this->set_output_dims(output_dims);
}

template <typename T, data_layout L, El::Device D>
void matmul_layer<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_matmul();
  msg->set_transpose_a(m_transpose_a);
  msg->set_transpose_b(m_transpose_b);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::fp_compute()
{

#ifdef LBANN_HAS_DISTCONV
  // We are guaranteed to have
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().fp_compute();
    return;
  }
#endif // LBANN_HAS_DISTCONV

  fp_compute_impl(*this, m_transpose_a, m_transpose_b);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::bp_compute()
{
#ifdef LBANN_HAS_DISTCONV
  // We are guaranteed to have
  if (this->distconv_enabled()) {
    this->get_distconv_adapter().bp_compute();
    return;
  }

#endif // LBANN_HAS_DISTCONV

  bp_compute_impl(*this, m_transpose_a, m_transpose_b);
}

// Explicit instantiation
#define PROTO_DEVICE(T, Device)                                                \
  template class matmul_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
