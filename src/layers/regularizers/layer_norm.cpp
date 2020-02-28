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

#define LBANN_LAYER_NORM_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/layer_norm.hpp"

namespace lbann {

namespace {

/** @brief Forward prop */
void fp_impl(lbann_comm& comm,
             DataType epsilon,
             const AbsDistMat& input,
             AbsDistMat& output,
             AbsDistMat& statistics) {

  // Local matrices
  const auto& local_input = dynamic_cast<const CPUMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<CPUMat&>(output.Matrix());
  auto& local_statistics = dynamic_cast<CPUMat&>(statistics.Matrix());
  auto local_means = El::LockedView(local_statistics, El::IR(0), El::ALL);
  auto local_vars = El::LockedView(local_statistics, El::IR(1), El::ALL);

  // Dimensions
  const El::Int sample_size = input.Height();
  const El::Int local_num_samples = local_input.Width();
  const El::Int local_sample_size = local_input.Height();

  // Compute sums
  El::Zero(statistics);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_num_samples; ++i) {
    auto& sum = local_means(0,i);
    auto& sqsum = local_vars(0,i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      sum += x;
      sqsum += x * x;
    }
  }
  comm.allreduce(statistics, statistics.RedundantComm(), El::mpi::SUM);

  // Compute statistics from sums
  //   mean = sum(x_i) / n
  //   var = ( sum(x_i^2)/n - mean^2 ) * n/(n-1)
  if (sample_size <= 1) {
    // local_means already has correct values
    El::Fill(local_vars, DataType{1});
  }
  else {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int i = 0; i < local_num_samples; ++i) {
      const auto sum = local_means(0,i);
      const auto sqsum = local_vars(0,i);
      const auto& mean = sum / sample_size;
      const auto& sqmean = sqsum / sample_size;
      const auto& var = (sqmean - mean*mean) * sample_size / (sample_size-1);
      local_means(0,i) = mean;
      local_vars(0,i) = std::max(var, DataType{0});
    }
  }

  // Apply layer norm
  //   y_i = (x_i - mean) / sqrt(var + epsilon)
  for (El::Int i = 0; i < local_num_samples; ++i) {
    const auto& mean = local_means(0,i);
    const auto& var = local_vars(0,i);
    const DataType inv_stdev = 1 / std::sqrt(var + epsilon);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      auto& y = local_output(j,i);
      y = (x - mean) * inv_stdev;
    }
  }

}

/** @brief Backprop */
void bp_impl(lbann_comm& comm,
             DataType epsilon,
             const AbsDistMat& input,
             const AbsDistMat& output_grad,
             AbsDistMat& input_grad,
             const AbsDistMat& statistics,
             AbsDistMat& statistics_grad) {

  // Local matrices
  const auto& local_input = dynamic_cast<const CPUMat&>(input.LockedMatrix());
  const auto& local_output_grad = dynamic_cast<const CPUMat&>(output_grad.LockedMatrix());
  auto& local_input_grad = dynamic_cast<CPUMat&>(input_grad.Matrix());
  const auto& local_statistics = dynamic_cast<const CPUMat&>(statistics.LockedMatrix());
  const auto local_means = El::LockedView(local_statistics, El::IR(0), El::ALL);
  const auto local_vars = El::LockedView(local_statistics, El::IR(1), El::ALL);
  auto& local_statistics_grad = dynamic_cast<CPUMat&>(statistics_grad.Matrix());
  auto local_means_grad = El::View(local_statistics_grad, El::IR(0), El::ALL);
  auto local_vars_grad = El::View(local_statistics_grad, El::IR(1), El::ALL);

  // Dimensions
  const El::Int sample_size = input.Height();
  const El::Int local_num_samples = local_input.Width();
  const El::Int local_sample_size = local_input.Height();

  // Trivial case if sample size <= 1
  // Note: Output is constant, so error signal is zero.
  if (sample_size <= 1) {
    El::Zero(input_grad);
    return;
  }

  // Compute gradient w.r.t. statistics
  //   dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
  //   dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
  El::Zero(statistics_grad);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_num_samples; ++i) {
    const auto& mean = local_means(0,i);
    const auto& var = local_vars(0,i);
    const DataType inv_stdev = 1 / std::sqrt(var + epsilon);
    auto& dmean = local_means_grad(0,i);
    auto& dvar = local_vars_grad(0,i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      const auto& dy = local_output_grad(j,i);
      dmean += dy;
      dvar += dy * (x - mean);
    }
    dmean *= -inv_stdev;
    dvar *= -inv_stdev*inv_stdev*inv_stdev / 2;
  }
  comm.allreduce(statistics_grad,
                 statistics_grad.RedundantComm(),
                 El::mpi::SUM);

  // Compute gradient w.r.t. input
  //   dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
  //             + dL/dmean / n
  //             + dL/dvar * (x_i - mean) * 2/(n-1) )
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_num_samples; ++i) {
    const auto& mean = local_means(0,i);
    const auto& var = local_vars(0,i);
    const DataType inv_stdev = 1 / std::sqrt(var + epsilon);
    const auto& dmean = local_means_grad(0,i);
    const auto& dvar = local_vars_grad(0,i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      const auto& dy = local_output_grad(j,i);
      auto& dx = local_input_grad(j,i);
      dx = (dy * inv_stdev
            + dmean / sample_size
            + dvar * (x - mean) * 2 / (sample_size - 1));
    }
  }

}

} // namespace <anon>

// Template instantiation
template <>
void layer_norm_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_activations(),
          *m_statistics);
}
template <>
void layer_norm_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_activations(),
          *m_statistics);
}
template <>
void layer_norm_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_prev_error_signals(),
          get_error_signals(),
          *m_statistics,
          *m_statistics_gradient);
}
template <>
void layer_norm_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_prev_error_signals(),
          get_error_signals(),
          *m_statistics,
          *m_statistics_gradient);
}

template class layer_norm_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
template class layer_norm_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;

} // namespace lbann
