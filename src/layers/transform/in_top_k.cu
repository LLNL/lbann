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
#include "lbann/utils/exception.hpp"

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
  __host__ __device__ static bool compare(const entry& a, const entry& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }

};

/** CPU implementation of in_top_k layer forward prop. */
void fp_gpu(lbann_comm& comm,
            El::Int k, const AbsDistMat& input, AbsDistMat& output) {
  LBANN_ERROR("not yet implemented");
}

} // namespace

template <>
void in_top_k_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(), m_k, get_prev_activations(), get_activations());
}
template <>
void in_top_k_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(), m_k, get_prev_activations(), get_activations());
}

} // namespace lbann
