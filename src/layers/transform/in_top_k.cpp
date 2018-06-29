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

#include "lbann/layers/transform/in_top_k.hpp"
#include <algorithm>
#include <limits>


namespace lbann {

namespace {

/** Sparse vector entry. */
struct entry {

  /** Vector entry value. */
  DataType value = -std::numeric_limits<DataType>::infinity();
  /** Vector entry index. */
  El::Int index = std::numeric_limits<El::Int>::max();

  /** Comparison operation to sort vector entries.
   *  Entries are sorted by value in decreasing order. Entries with
   *  the same value are sorted by index in increasing order.
   */
  static bool compare(const entry& a, const entry& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }

};

/** CPU implementation of in_top_k layer forward prop. */
void fp_cpu(lbann_comm& comm,
            El::Int k, const AbsDistMat& input, AbsDistMat& output) {

  // Trivial cases
  if (k < 1) {
    El::Zero(output);
    return;
  } else if (k >= input.Height()) {
    El::Fill(output, DataType(1));
    return;
  }
  
  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Find top-k entries in each local matrix column
  std::vector<entry> top_k_entries(k * local_width);
#pragma omp parallel for
  for (El::Int col = 0; col < local_width; ++col) {
    std::vector<entry> local_entries(std::max(local_height, k));
    for (El::Int row = 0; row < local_height; ++row) {
      local_entries[row].value = local_input(row, col);
      local_entries[row].index = input.GlobalRow(row);
    }
    std::partial_sort_copy(local_entries.begin(),
                           local_entries.end(),
                           &top_k_entries[k*col],
                           &top_k_entries[k*(col+1)],
                           entry::compare);
  }
  
  // Find top-k entries in each global matrix column
  // Note: Nothing needs to be done if matrix columns are not
  // distributed.
  auto&& col_comm = input.ColComm();
  const El::Int col_comm_size = El::mpi::Size(col_comm);
  if (col_comm_size > 1 && local_width > 0) {
    std::vector<entry> global_top_k_entries(k * local_width * col_comm_size);
    comm.all_gather(reinterpret_cast<El::byte*>(top_k_entries.data()),
                    top_k_entries.size() * sizeof(entry) / sizeof(El::byte),
                    reinterpret_cast<El::byte*>(global_top_k_entries.data()),
                    top_k_entries.size() * sizeof(entry) / sizeof(El::byte),
                    col_comm);
#pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      std::vector<entry> entries(col_comm_size * k);
      for (El::Int rank = 0; rank < col_comm_size; ++rank) {
        const auto& offset = rank * local_width * k + col * k;
        std::copy(&global_top_k_entries[offset],
                  &global_top_k_entries[offset+k],
                  &entries[k*rank]);
      }
      std::partial_sort_copy(entries.begin(),
                             entries.end(),
                             &top_k_entries[k*col],
                             &top_k_entries[k*(col+1)],
                             entry::compare);
    }
  }

  // Indicate output entries corresponding to top-k input entries
  El::Zero(output);
#pragma omp parallel for collapse(2)
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int i = 0; i < k; ++i) {
      const auto& global_row = top_k_entries[col*k+i].index;
      if (output.IsLocalRow(global_row)) {
        local_output(output.LocalRow(global_row), col) = DataType(1);
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
