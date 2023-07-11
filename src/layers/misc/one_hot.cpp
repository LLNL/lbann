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

#define LBANN_ONE_HOT_LAYER_INSTANTIATE
#include "lbann/layers/misc/one_hot_impl.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void one_hot_layer<TensorDataType, Layout, Device>::fp_compute()
{

  // Local matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& input = this->get_prev_activations();
  auto& output = this->get_activations();
  auto& local_output = dynamic_cast<LocalMat&>(output.Matrix());
  const El::Int local_mini_batch_size = output.LocalWidth();
  const El::Int output_size = output.Height();

  // Make sure all procs in column communicator have access to input
  LocalMat local_input;
  const auto& col_comm = input.ColComm();
  const auto col_rank = El::mpi::Rank(col_comm);
  const auto owner_rank = input.RowOwner(0);
  if (col_rank == owner_rank) {
    El::LockedView(local_input, input.LockedMatrix());
  }
  else {
    local_input.Resize(1, input.LocalWidth());
  }
  /** @todo (tym1 3/12/21): We are working around a bug in Hydrogen.
   *  Broadcast with Matrix<T,D> is not instatiated. */
  El::Broadcast(static_cast<El::AbstractMatrix<TensorDataType>&>(local_input),
                col_comm,
                owner_rank);

  // Populate one-hot vectors
  El::Zero(output);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int j = 0; j < local_mini_batch_size; ++j) {
    const auto& x = local_input.CRef(0, j);
    const auto i_global = static_cast<El::Int>(std::floor(x));
    if (0 <= i_global && i_global < output_size &&
        output.RowOwner(i_global) == col_rank) {
      local_output(output.LocalRow(i_global), j) =
        El::TypeTraits<TensorDataType>::One();
    }
  }
}

#define PROTO(T)                                                               \
  template class one_hot_layer<T,                                              \
                               data_layout::DATA_PARALLEL,                     \
                               El::Device::CPU>;                               \
  template class one_hot_layer<T, data_layout::MODEL_PARALLEL, El::Device::CPU>
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
