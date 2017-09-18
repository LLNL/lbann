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

#ifndef LBANN_METRIC_TOP_K_CATEGORICAL_ACCURACY_HPP
#define LBANN_METRIC_TOP_K_CATEGORICAL_ACCURACY_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

namespace metrics {

/**
 * A prediction is correct under this metric if any of the top k predictions
 * are the correct prediction.
 */
template <data_layout T_layout>
class top_k_categorical_accuracy : public metric {
 public:
  top_k_categorical_accuracy(int top_k,
                             lbann_comm *comm) :
    metric(comm), m_top_k(top_k) {}

  top_k_categorical_accuracy(
    const top_k_categorical_accuracy<T_layout>& other) = default;
  top_k_categorical_accuracy& operator=(
    const top_k_categorical_accuracy<T_layout>& other) = default;

  ~top_k_categorical_accuracy() {}

  top_k_categorical_accuracy* copy() const {
    return new top_k_categorical_accuracy(*this);
  }

  void setup(int num_neurons, int mini_batch_size) {
    metric::setup(num_neurons, mini_batch_size);
  }
  void fp_set_std_matrix_view(int cur_mini_batch_size) {}
  double compute_metric(AbsDistMat& predictions_v, AbsDistMat& ground_truth_v) {
    // This first computes the top k predictions within each column locally,
    // then each column master gathers these, computes the global top k, and
    // determines if an error was made.
    El::Int num_errors = 0;
    // Note: assumes structure is packed.
    struct top_k_ele {
      DataType val;  // Predicted value.
      DataType gt;  // Ground truth.
    };
    const El::Int local_width = predictions_v.LocalWidth();  // minibatch dim
    const El::Int local_height = predictions_v.LocalHeight();  // class dim
    // Pack the top k predictions for each local column together.
    std::vector<top_k_ele> local_top_k(m_top_k * local_width);
    // Compute the top k entries locally.
    std::vector<El::Int> local_indices(local_height);
    std::iota(local_indices.begin(), local_indices.end(), 0);
    for (El::Int mb_idx = 0; mb_idx < local_width; ++mb_idx) {
      // Determine the top k local entries in this column.
      std::partial_sort(
        local_indices.begin(), local_indices.begin() + m_top_k,
        local_indices.end(),
        [mb_idx, &predictions_v, this] (El::Int a, El::Int b) -> bool {
          return predictions_v.GetLocal(a, mb_idx) >
            predictions_v.GetLocal(b, mb_idx); });
      for (El::Int i = 0; i < m_top_k; ++i) {
        El::Int idx = mb_idx*m_top_k + i;
        local_top_k[idx].val = predictions_v.GetLocal(local_indices[i], mb_idx);
        local_top_k[idx].gt = ground_truth_v.GetLocal(local_indices[i], mb_idx);
      }
    }
    // Gather the data for each column to rank 0 within that column.
    El::mpi::Comm col_comm = predictions_v.ColComm();
    int col_comm_size = El::mpi::Size(col_comm);
    if (El::mpi::Rank(col_comm) == 0) {
      // This vector ends up being the concatenation of each local_top_k, and
      // therefore accessing data for a single mini-batch requires computing the
      // appropriate strides.
      std::vector<top_k_ele> global_top_k(
        m_top_k * local_width * col_comm_size);
      El::mpi::Gather((DataType*) local_top_k.data(), 2*local_top_k.size(),
                      (DataType*) global_top_k.data(), 2*local_top_k.size(),
                      0, col_comm);
      // Compute the global top k elements in each column.
      std::vector<El::Int> global_indices(m_top_k * col_comm_size);
      std::iota(global_indices.begin(), global_indices.end(), 0);
      for (El::Int mb_idx = 0; mb_idx < local_width; ++mb_idx) {
        std::partial_sort(
          global_indices.begin(), global_indices.begin() + m_top_k,
          global_indices.end(),
          [mb_idx, col_comm_size, &global_top_k, this]
          (El::Int a, El::Int b) -> bool {
            El::Int mb_offset = mb_idx * m_top_k;
            El::Int a_proc_offset = (a/m_top_k) * m_top_k * col_comm_size;
            El::Int a_idx = a_proc_offset + mb_offset + (a%m_top_k);
            El::Int b_proc_offset = (b/m_top_k) * m_top_k * col_comm_size;
            El::Int b_idx = b_proc_offset + mb_offset + (b%m_top_k);
            return global_top_k[a_idx].val > global_top_k[b_idx].val;
          });
        // Check if there is a 1 ground truth label in the top k.
        bool found = false;
        for (El::Int i = 0; i < m_top_k; ++i) {
          El::Int idx = global_indices[i];
          idx = mb_idx*m_top_k + (i/m_top_k)*m_top_k*col_comm_size + (i%m_top_k);
          if (global_top_k[idx].gt == DataType(1)) {
            found = true;
            break;
          }
        }
        if (!found) {
          ++num_errors;
        }
      }
    } else {
      El::mpi::Gather((DataType*) local_top_k.data(), 2*local_top_k.size(),
                      (DataType*) NULL, 0, 0, col_comm);
    }
    return this->m_comm->model_allreduce(num_errors);
  }

  double report_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double errors_per_epoch = stats->m_error_per_epoch;
    long samples_per_epoch = stats->m_samples_per_epoch;

    double accuracy = (double)(samples_per_epoch - errors_per_epoch) /
      samples_per_epoch * 100;
    string score = std::to_string(accuracy);

    return accuracy;
  }
  double report_lifetime_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double total_error = stats->m_total_error;
    long total_num_samples = stats->m_total_num_samples;

    double accuracy = (double)(total_num_samples - total_error) / total_num_samples * 100;
    string score = std::to_string(accuracy);

    return accuracy;
  }

  std::string name() const {
    return "top-" + std::to_string(m_top_k) + " categorical accuracy";
  }
  std::string display_unit() const { return "%"; }

 protected:
  /** Number of top classes to check for correct prediction. */
  int m_top_k;
  /** Maximum minibatch size this metric will be used with. */
  int64_t m_max_mini_batch_size;
};

}  // namespace metrics

}  // namespace lbann

#endif // LBANN_METRIC_TOP_K_CATEGORICAL_ACCURACY_HPP
