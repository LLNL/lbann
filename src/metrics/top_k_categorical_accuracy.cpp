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

#include "lbann/metrics/top_k_categorical_accuracy.hpp"

namespace lbann {

top_k_categorical_accuracy_metric::top_k_categorical_accuracy_metric(int top_k,
                                                                     lbann_comm *comm) 
  : metric(comm), m_top_k(top_k) {}

DataType top_k_categorical_accuracy_metric::evaluate_compute(const AbsDistMat& prediction,
                                                             const AbsDistMat& ground_truth) {
    // This first computes the top k predictions within each column locally,
    // then each column master gathers these, computes the global top k, and
    // determines if an error was made.
    El::Int num_errors = 0;
    // Note: assumes structure is packed.
    struct top_k_ele {
      DataType val;  // Predicted value.
      DataType gt;  // Ground truth.
    };
    const El::Int local_width = prediction.LocalWidth();  // minibatch dim
    const El::Int local_height = prediction.LocalHeight();  // class dim
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
        [mb_idx, &prediction] (El::Int a, El::Int b) -> bool {
          return prediction.GetLocal(a, mb_idx) >
            prediction.GetLocal(b, mb_idx); });
      for (El::Int i = 0; i < m_top_k; ++i) {
        El::Int idx = mb_idx*m_top_k + i;
        local_top_k[idx].val = prediction.GetLocal(local_indices[i], mb_idx);
        local_top_k[idx].gt = ground_truth.GetLocal(local_indices[i], mb_idx);
      }
    }
    // Gather the data for each column to rank 0 within that column.
    El::mpi::Comm col_comm = prediction.ColComm();
    int col_comm_size = El::mpi::Size(col_comm);
    if (El::mpi::Rank(col_comm) == 0) {
      // This vector ends up being the concatenation of each local_top_k, and
      // therefore accessing data for a single mini-batch requires computing the
      // appropriate strides.
      std::vector<top_k_ele> global_top_k(
        m_top_k * local_width * col_comm_size);
      get_comm().gather((DataType*) local_top_k.data(), 2*local_top_k.size(),
                     (DataType*) global_top_k.data(), col_comm);
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
      get_comm().gather((DataType*) local_top_k.data(), 2*local_top_k.size(), 0,
                        col_comm);
    }
    num_errors = get_comm().model_allreduce(num_errors);
    const int mini_batch_size = prediction.Width();
    return (mini_batch_size - num_errors) * DataType(100) / mini_batch_size;
}

}  // namespace lbann
