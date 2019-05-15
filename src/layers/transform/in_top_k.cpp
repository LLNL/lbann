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

#include "lbann/layers/transform/in_top_k.hpp"
#include <algorithm>
#include <limits>


namespace lbann {

namespace {

/** Sparse vector entry. */
struct entry {

  /** Vector entry value. */
  DataType value = min_value;
  /** Vector entry index. */
  El::Int index = max_index;

  /** Minimum possible value. */
  static constexpr DataType min_value = -std::numeric_limits<DataType>::infinity();
  /** Maximum possible index. */
  static constexpr El::Int max_index = std::numeric_limits<El::Int>::max();

  /** Comparison operation to sort vector entries.
   *  Entries are sorted by value in decreasing order, with ties
   *  broken in favor of entries with smaller indices.
   */
  static bool compare(const entry& a, const entry& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }

};

/** CPU implementation of in_top_k layer forward prop. */
void fp_cpu(lbann_comm& comm,
            El::Int k,
            const AbsDistMat& input,
            AbsDistMat& output) {

  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  const El::Int height = input.Height();
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Trivial cases
  if (k < 1) {
    El::Zero(output);
    return;
  } else if (k >= height) {
    El::Fill(output, DataType(1));
    return;
  } else if (local_width < 1) {
    return;
  }

  // Column communicator
  auto&& col_comm = input.ColComm();
  const auto& col_comm_size = El::mpi::Size(col_comm);

  // Find top-k entries in each column of local input matrix
  std::vector<entry> top_entries(local_width * k);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    std::vector<entry> local_entries(std::max(local_height, k));
    for (El::Int row = 0; row < local_height; ++row) {
      local_entries[row].value = local_input(row, col);
      local_entries[row].index = input.GlobalRow(row);
    }
    std::partial_sort_copy(local_entries.begin(),
                           local_entries.end(),
                           &top_entries[col*k],
                           &top_entries[col*k] + k,
                           entry::compare);
  }

  // Find top-k entries in each column of global input matrix
  if (col_comm_size > 1) {
    std::vector<entry> global_top_entries(col_comm_size * local_width * k);
    comm.all_gather(reinterpret_cast<El::byte*>(top_entries.data()),
                    top_entries.size() * sizeof(entry),
                    reinterpret_cast<El::byte*>(global_top_entries.data()),
                    top_entries.size() * sizeof(entry),
                    col_comm);
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {
      std::vector<entry> col_entries(col_comm_size * k);
      for (El::Int rank = 0; rank < col_comm_size; ++rank) {
        const auto* start = &global_top_entries[rank*local_width*k+col*k];
        std::copy(start, start + k, &col_entries[rank*k]);
      }
      std::partial_sort_copy(col_entries.begin(),
                             col_entries.end(),
                             &top_entries[col*k],
                             &top_entries[col*k] + k,
                             entry::compare);
    }
  }

  // Indicate output entries corresponding to top-k input entries
  El::Zero(output);
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int i = 0; i < k; ++i) {
      const auto& global_row = top_entries[col*k+i].index;
      if (global_row < height && output.IsLocalRow(global_row)) {
        const auto& row = output.LocalRow(global_row);
        local_output(row, col) = DataType(1);
      }
    }
  }

}

} // namespace

template <>
void in_top_k_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::fp_compute() {
  fp_cpu(*get_comm(), m_k, get_prev_activations(), get_activations());
}
template <>
void in_top_k_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::fp_compute() {
  fp_cpu(*get_comm(), m_k, get_prev_activations(), get_activations());
}

} // namespace lbann
