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

#ifndef LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
#define LBANN_METRIC_CATEGORICAL_ACCURACY_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

namespace metrics {

template <data_layout T_layout>
class categorical_accuracy : public metric {
 protected:
  // These are to simplify implementation below.
  typedef typename std::conditional<T_layout == data_layout::MODEL_PARALLEL,
                                    ColSumMat,
                                    // else, DATA_PARALLEL:
                                    ColSumStarVCMat>::type colmat_t;
 public:
  /// Constructor
  categorical_accuracy(lbann_comm *comm)
    : metric(comm),
      m_prediction_col_maxes(comm->get_model_grid()),
      m_replicated_prediction_col_maxes(comm->get_model_grid()),
      m_prediction_col_maxes_v(comm->get_model_grid()),
      m_replicated_prediction_col_maxes_v(comm->get_model_grid()) {}

  categorical_accuracy(const categorical_accuracy<T_layout>& other) = default;

  categorical_accuracy& operator=(
    const categorical_accuracy<T_layout>& other) = default;

  /// Destructor
  ~categorical_accuracy() {}

  categorical_accuracy* copy() const { return new categorical_accuracy(*this); }

  void setup(int num_neurons, int mini_batch_size) {
    metric::setup(num_neurons, mini_batch_size);
    // Clear the contents of the intermediate matrices
    El::Zeros(m_prediction_col_maxes, mini_batch_size, 1);
    El::Zeros(m_replicated_prediction_col_maxes, mini_batch_size, 1);
    El::Zeros(m_max_index, mini_batch_size, 1); // Clear the entire matrix
    El::Zeros(m_reduced_max_indices, mini_batch_size, 1); // Clear the entire matrix
    m_max_mini_batch_size = mini_batch_size;
  }

  void fp_set_std_matrix_view(int cur_mini_batch_size) {
    // Set the view based on the size of the current mini-batch
    // Note that these matrices are transposed (column max matrices) and thus
    // the mini-batch size effects the number of rows, not columns
    El::View(m_prediction_col_maxes_v, m_prediction_col_maxes,
             El::IR(0, cur_mini_batch_size), El::IR(0, m_prediction_col_maxes.Width()));
    El::View(m_replicated_prediction_col_maxes_v, m_replicated_prediction_col_maxes,
             El::IR(0, cur_mini_batch_size),
             El::IR(0, m_replicated_prediction_col_maxes.Width()));
    El::View(m_max_index_v, m_max_index, El::IR(0, cur_mini_batch_size),
             El::IR(0, m_max_index.Width()));
    El::View(m_reduced_max_indices_v, m_reduced_max_indices,
             El::IR(0, cur_mini_batch_size), El::IR(0, m_reduced_max_indices.Width()));
  }

  double compute_metric(AbsDistMat& predictions_v, AbsDistMat& groundtruth_v) {
    // Clear the contents of the intermediate matrices
    El::Zeros(m_prediction_col_maxes, m_max_mini_batch_size, 1);
    El::Zeros(m_replicated_prediction_col_maxes, m_max_mini_batch_size, 1);

    /// Compute the error between the previous layers activations and the ground truth
    /// For each minibatch (column) find the maximimum value
    typedef typename std::conditional<T_layout == data_layout::MODEL_PARALLEL,
                                      DistMat, StarVCMat>::type colnorm_cast_t;
    ColumnMaxNorms((colnorm_cast_t) predictions_v, m_prediction_col_maxes_v);
    El::Copy(m_prediction_col_maxes_v, m_replicated_prediction_col_maxes_v); /// Give every rank a copy so that they can find the max index locally

    El::Zeros(m_max_index, m_max_mini_batch_size, 1); // Clear the entire matrix

    /// Find which rank holds the index for the maxmimum value
    for(int mb_index = 0; mb_index < predictions_v.LocalWidth(); mb_index++) { /// For each sample in mini-batch that this rank has
      int mb_global_index = predictions_v.GlobalCol(mb_index);
      DataType sample_max = m_replicated_prediction_col_maxes_v.GetLocal(mb_global_index, 0);
      for(int f_index = 0; f_index < predictions_v.LocalHeight(); f_index++) { /// For each feature
        if(predictions_v.GetLocal(f_index, mb_index) == sample_max) {
          m_max_index_v.Set(mb_global_index, 0, predictions_v.GlobalRow(f_index));
        }
      }
    }

    El::Zeros(m_reduced_max_indices, m_max_mini_batch_size, 1); // Clear the entire matrix
    /// Merge all of the local index sets into a common buffer, if there are two potential maximum values, highest index wins
    /// Note that this has to operate on the raw buffer, not the view
    m_comm->model_allreduce(m_max_index.Buffer(),
                            m_max_index.Height() * m_max_index.Width(),
                            m_reduced_max_indices.Buffer(), mpi::MAX);

    /// Check to see if the predicted results match the target results
    int num_errors = 0;

    //  Copy(groundtruth_v, Y_local);

    /// @todo - BVE I believe that the following code works, but doesn't
    /// need to be this hard  it shouldn't have to check the inequality
    /// since there should only be one category turned on in the ground truth

    /// Distributed search over the groundtruth matrix
    /// Each rank will search its local portion of the matrix to find if it has the true category
    for(int mb_index= 0; mb_index < groundtruth_v.LocalWidth(); mb_index++) { /// For each sample in mini-batch
      int targetidx = -1;
      for(int f_index= 0; f_index < groundtruth_v.LocalHeight(); f_index++) {
        if(groundtruth_v.GetLocal(f_index, mb_index) == DataType(1)) {
          targetidx = groundtruth_v.GlobalRow(f_index); /// If this rank holds the correct category, return the global row index
        }
      }
      if(targetidx != -1) { /// Only check against the prediction if this rank holds the groundtruth value
        int global_mb_index = groundtruth_v.GlobalCol(mb_index);
        if(m_reduced_max_indices_v.Get(global_mb_index, 0) != targetidx) {
          num_errors++;
        }
      }
    }

    num_errors = m_comm->model_allreduce(num_errors);
    return num_errors;
  }

  double report_metric(execution_mode mode) {
    statistics *stats = get_statistics(mode);
    double errors_per_epoch = stats->m_error_per_epoch;
    long samples_per_epoch = stats->m_samples_per_epoch;

    double accuracy = (double)(samples_per_epoch - errors_per_epoch) / samples_per_epoch * 100;
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

  std::string name() const { return "categorical accuracy"; }
  std::string display_unit() const { return "%"; }

 protected:
  /// The maximum value within each minibatch (column).
  colmat_t m_prediction_col_maxes;
  /// m_prediction_col_maxes, replicated onto every rank.
  StarMat m_replicated_prediction_col_maxes;
  Mat m_max_index;    /// Local array to hold max indicies
  Mat m_reduced_max_indices;  /// Local array to build global view of maximum indicies

  /// View of m_prediction_col_maxes for the current minibatch size.
  colmat_t m_prediction_col_maxes_v;
  /// View of m_replicated_prediction_col_maxes for the current minibatch size.
  StarMat m_replicated_prediction_col_maxes_v;
  Mat m_max_index_v;
  Mat m_reduced_max_indices_v;

  int m_max_mini_batch_size;
};

}  // namespace metrics

}  // namespace lbann


#endif  // LBANN_METRIC_CATEGORICAL_ACCURACY_HPP
