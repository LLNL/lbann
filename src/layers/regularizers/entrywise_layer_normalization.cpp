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

#define LBANN_ENTRYWISE_LAYER_NORMALIZATION_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/entrywise_layer_normalization.hpp"

namespace lbann {

namespace {

void fp_impl(lbann_comm& comm,
             DataType epsilon,
             const AbsDistMat& input,
             AbsDistMat& output,
             AbsDistMat& statistics) {

  // Local matrices
  const auto& local_input = dynamic_cast<const CPUMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<CPUMat&>(output.Matrix());
  auto& local_statistics = dynamic_cast<CPUMat&>(statistics.Matrix());
  auto local_mean = El::LockedView(local_statistics, El::IR(0), El::ALL);
  auto local_var = El::LockedView(local_statistics, El::IR(1), El::ALL);

  // Dimensions
  const El::Int sample_size = input.Height();
  const El::Int local_mini_batch_size = local_input.Width();
  const El::Int local_sample_size = local_input.Height();

  // Compute sums
  El::Zero(statistics);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    auto& sum = local_mean(0,i);
    auto& sqsum = local_var(0,i);
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
    // local_mean already has correct values
    El::Fill(local_var, DataType{1});
  } else {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int i = 0; i < local_mini_batch_size; ++i) {
      const auto sum = local_mean(0,i);
      const auto sqsum = local_var(0,i);
      const auto& mean = sum / sample_size;
      const auto& sqmean = sqsum / sample_size;
      const auto& var = (sqmean - mean*mean) * sample_size / (sample_size-1);
      local_mean(0,i) = mean;
      local_var(0,i) = std::max(var, DataType{0});
    }
  }

  // Apply layer norm
  //   y_i = (x_i - mean) / sqrt(var + epsilon)
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    const auto& mean = local_mean(0,i);
    const auto& var = local_var(0,i);
    const DataType inv_stdev = 1 / std::sqrt(var + epsilon);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      auto& y = local_output(j,i);
      y = (x - mean) * inv_stdev;
    }
  }

}

void bp_impl(lbann_comm& comm,
             DataType epsilon,
             const AbsDistMat& input,
             const AbsDistMat& gradient_wrt_output,
             AbsDistMat& gradient_wrt_input,
             const AbsDistMat& statistics,
             AbsDistMat& gradient_wrt_statistics) {

  // Local matrices
  const auto& local_input = dynamic_cast<const CPUMat&>(input.LockedMatrix());
  const auto& local_gradient_wrt_output = dynamic_cast<const CPUMat&>(gradient_wrt_output.LockedMatrix());
  auto& local_gradient_wrt_input = dynamic_cast<CPUMat&>(gradient_wrt_input.Matrix());
  const auto& local_statistics = dynamic_cast<const CPUMat&>(statistics.LockedMatrix());
  const auto local_mean = El::LockedView(local_statistics, El::IR(0), El::ALL);
  const auto local_var = El::LockedView(local_statistics, El::IR(1), El::ALL);
  auto& local_gradient_wrt_statistics = dynamic_cast<CPUMat&>(gradient_wrt_statistics.Matrix());
  auto local_gradient_wrt_mean = El::View(local_gradient_wrt_statistics, El::IR(0), El::ALL);
  auto local_gradient_wrt_var = El::View(local_gradient_wrt_statistics, El::IR(1), El::ALL);

  // Dimensions
  const El::Int sample_size = input.Height();
  const El::Int local_mini_batch_size = local_input.Width();
  const El::Int local_sample_size = local_input.Height();

  // Trivial case if sample size <= 1
  // Note: Output is constant, so error signal is zero.
  if (sample_size <= 1) {
    El::Zero(gradient_wrt_input);
    return;
  }

  // Compute gradient w.r.t. statistics
  //   dL/dmean = - sum(dL/dy_i) / sqrt(var+epsilon)
  //   dL/dvar = - sum(dL/dy_i * (x_i-mean)) * (var+epsilon)^(-3/2) / 2
  El::Zero(gradient_wrt_statistics);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    const auto& mean = local_mean(0,i);
    const auto& var = local_var(0,i);
    const DataType inv_stdev = 1 / std::sqrt(var + epsilon);
    auto& dmean = local_gradient_wrt_mean(0,i);
    auto& dvar = local_gradient_wrt_var(0,i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      const auto& dy = local_gradient_wrt_output(j,i);
      dmean += dy;
      dvar += dy * (x - mean);
    }
    dmean *= -inv_stdev;
    dvar *= -inv_stdev*inv_stdev*inv_stdev / 2;
  }
  comm.allreduce(gradient_wrt_statistics,
                 gradient_wrt_statistics.RedundantComm(),
                 El::mpi::SUM);

  // Compute gradient w.r.t. input
  //   dL/dx_i = ( dL/dy_i / sqrt(var+epsilon)
  //             + dL/dmean / n
  //             + dL/dvar * (x_i - mean) * 2/(n-1) )
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    const auto& mean = local_mean(0,i);
    const auto& var = local_var(0,i);
    const DataType inv_stdev = 1 / std::sqrt(var + epsilon);
    const auto& dmean = local_gradient_wrt_mean(0,i);
    const auto& dvar = local_gradient_wrt_var(0,i);
    for (El::Int j = 0; j < local_sample_size; ++j) {
      const auto& x = local_input(j,i);
      const auto& dy = local_gradient_wrt_output(j,i);
      auto& dx = local_gradient_wrt_input(j,i);
      dx = (dy * inv_stdev
            + dmean / sample_size
            + dvar * (x - mean) * 2 / (sample_size - 1));
    }
  }

}

} // namespace <anon>

// Template instantiation
template <>
void entrywise_layer_normalization_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_activations(),
          *m_statistics);
}
template <>
void entrywise_layer_normalization_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_activations(),
          *m_statistics);
}
template <>
void entrywise_layer_normalization_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_prev_error_signals(),
          get_error_signals(),
          *m_statistics,
          *m_statistics_gradient);
}
template <>
void entrywise_layer_normalization_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_impl(*get_comm(),
          m_epsilon,
          get_prev_activations(),
          get_prev_error_signals(),
          get_error_signals(),
          *m_statistics,
          *m_statistics_gradient);
}

template class entrywise_layer_normalization_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
template class entrywise_layer_normalization_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;

} // namespace lbann
