///////////////////////////////////////////////////////////////////////////////
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

#ifndef LBANN_LAYER_LOGSOFTMAX_HPP_INCLUDED
#define LBANN_LAYER_LOGSOFTMAX_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/cudnn.hpp"
#include <unistd.h>
#include <string>

#include <cassert>

namespace lbann {

#ifdef LBANN_HAS_GPU
namespace logsoftmax_cuda {
/** Compute the maximum entry in input for each column.
 * Data is assumed to be on the GPU.
 */
void max_local_col_entry(int height, int width,
                         const DataType * __restrict__ input,
                         int input_ldim,
                         DataType * __restrict__ workspace,
                         cudaStream_t stream);
/** Exponentiate the (shifted) input, and compute its sum.
 * Data is assumed to be on the GPU.
 */
void exp_and_col_sum(int height, int width,
                     const DataType * __restrict__ input,
                     int input_ldim,
                     DataType * __restrict__ output,
                     int output_ldim,
                     DataType * __restrict__ workspace,
                     cudaStream_t stream);
/** Subtract from each entry in a column the log of the pre-computed sum of the column.
 * Shift by the max column entry for the log-sum-exp trick.
 * Data is assumed to be on the GPU.
 */
void sub_by_col_sums_and_shift(int height, int width,
                                const DataType * __restrict__ input,
                                int input_ldim,
                                DataType * __restrict__ output,
                                int output_ldim,
                                const DataType * __restrict__ workspace,
                                cudaStream_t stream);
/** Compute column sums for the gradient w.r.t. the output.
 * Data is assumed to be on the GPU.
 */
void out_grad_col_sum(int height, int width,
                               DataType * __restrict__ workspace,
                               const DataType * __restrict__ grad_wrt_output,
                               int grad_wrt_output_ldim,
                               cudaStream_t stream);
/** Compute the gradient w.r.t. the input.
 * Data is assumed to be on the GPU.
 */
void grad_wrt_input(int height, int width,
                               const DataType * __restrict__ output,
                               int output_ldim,
                               const DataType * __restrict__ workspace,
                               const DataType * __restrict__ grad_wrt_output,
                               int grad_wrt_output_ldim,
                               DataType * __restrict__ grad_wrt_input,
                               int grad_wrt_input_ldim,
                               cudaStream_t stream);
}  // namespace logsoftmax_cuda
#endif // LBANN_HAS_CUDA

/** Softmax layer. */
template <data_layout T_layout, El::Device Dev>
class logsoftmax_layer : public activation_layer {

 private:

  /** Workspace for column-wise reductions. */
  AbsDistMat *m_workspace;

#ifdef LBANN_HAS_CUDNN
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

 public:

  logsoftmax_layer(lbann_comm *comm)
    : activation_layer(comm),
      m_workspace(nullptr)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {}

  logsoftmax_layer(const logsoftmax_layer& other)
    : activation_layer(other)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {

    // Matrix deep copy
    m_workspace = other.m_workspace;
    if (m_workspace != nullptr) { m_workspace = m_workspace->Copy(); }

#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN

  }

  logsoftmax_layer& operator=(const logsoftmax_layer& other) {
    activation_layer::operator=(other);

    // Deep matrix copy
    if (m_workspace != nullptr) { delete m_workspace; }
    m_workspace = other.m_workspace;
    if (m_workspace != nullptr) { m_workspace = m_workspace->Copy(); }

#ifdef LBANN_HAS_CUDNN
    // Copy cuDNN objects
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN

  }

  ~logsoftmax_layer() {
    if (m_workspace != nullptr) { delete m_workspace; }
  }

  logsoftmax_layer* copy() const override { return new logsoftmax_layer(*this); }
  std::string get_type() const override { return "logsoftmax"; }

  std::string get_description() const override {
    return std::string {} + " logsoftmax" + " dataLayout: "
           + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_matrices(const El::Grid& grid) override;

  void setup_data() override {
    activation_layer::setup_data();
    const int mini_batch_size = this->m_model->get_max_mini_batch_size();
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    activation_layer::fp_setup_outputs(mini_batch_size);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

  virtual void fp_compute_cpu() {

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    auto& local_workspace = m_workspace->Matrix();

    // Matrix parameters
    const El::Int local_height = local_input.Height();
    const El::Int local_width = local_input.Width();

    // Find maximum entry in each column
    if (local_height == 0) {
      // When there's no local data, fill the workspace with a small value so
      // the maximum across processors is still computed correctly.
      El::Fill(local_workspace, std::numeric_limits<DataType>::lowest());
    } else {
      LBANN_OMP_TASKLOOP
      for (El::Int col = 0; col < local_width; ++col) {
        DataType max_entry = local_input(0, col);
        for (El::Int row = 1; row < local_height; ++row) {
          max_entry = std::max(max_entry, local_input(row, col));
        }
        local_workspace(0, col) = max_entry;
      }
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(),
                      El::mpi::MAX);

    // Exponentiate activations and compute column sums
    // Note: Subtracting by the column max prevents activations from
    // blowing up. Large negative values underflow to 0.
    // Save the sum, shift values for the log-sum-exp trick.
    if (local_height == 0) {
      // When there's no local data, fill the workspace with zeros so
      // the sum across processors is still computed correctly.
      El::Fill(local_workspace, DataType(0));
    } else {
      LBANN_OMP_TASKLOOP
      for (El::Int col = 0; col < local_width; ++col) {
        const DataType shift = local_workspace(0, col);
        local_output(0, col) = shift;
        DataType sum = 0;
        for (El::Int row = 0; row < local_height; ++row) {
          const DataType x = local_input(row, col);
          const DataType y = std::exp(x - shift);
          sum += y;
        }
        local_workspace(0, col) = sum;
      }
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());

    // Subtract log-sum-exp and shift value from input to get numerically stable output.
    if (local_height > 0) {
      LBANN_OMP_TASKLOOP
      for (El::Int col = 0; col < local_width; ++col) {
        const DataType lse = std::log(local_workspace(0, col));
        const DataType shift = local_output(0, col);
        for (El::Int row = 0; row < local_height; ++row) {
          const DataType x = local_input(row, col);
          DataType& y = local_output(row, col);
          y = x - lse - shift;
        }
      }
    }
  }

  virtual void bp_compute_cpu() {

    // Local matrices
    const DMat<Dev>& local_output = get_local_activations();
    const DMat<Dev>& local_gradient_wrt_output = get_local_prev_error_signals();
    DMat<Dev>& local_gradient_wrt_input = get_local_error_signals();
    DMat<Dev>& local_workspace = m_workspace->Matrix();

    // Matrix parameters
    const El::Int local_height = local_output.Height();
    const El::Int local_width = local_output.Width();

    // Compute column sums for gradient w.r.t. output.
    LBANN_OMP_TASKLOOP
    for (El::Int col = 0; col < local_width; ++col) {
      DataType sum = 0;
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType dy = local_gradient_wrt_output(row, col);
        sum += dy;
      }
      local_workspace(0, col) = sum;
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());

    // Compute gradient w.r.t. input.
    LBANN_OMP_TASKLOOP
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType sum = local_workspace(0, col);
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType y = local_output(row, col);
        const DataType dy = local_gradient_wrt_output(row, col);
        DataType dx = dy - std::exp(y) * sum;
        local_gradient_wrt_input(row, col) = dx;
      }
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_LOGSOFTMAX_HPP_INCLUDED
