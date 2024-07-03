////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/transform/tessellate.hpp"

#ifdef LBANN_HAS_CUTENSOR
#include "lbann/utils/cutensor_support.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void tessellate_layer<TensorDataType, T_layout, Dev>::bp_compute_cutensor(
  const std::vector<int>& input_dims,
  const std::vector<int>& output_dims,
  const El::AbstractDistMatrix<TensorDataType>& output_grad,
  El::AbstractMatrix<TensorDataType>& input_grad)
{
  LBANN_ASSERT(Dev == El::Device::GPU);

  // Constants
  const auto one = El::TypeTraits<TensorDataType>::One();
  const auto zero = El::TypeTraits<TensorDataType>::Zero();

  // Data matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  LocalMat const& input =
    static_cast<LocalMat const&>(output_grad.LockedMatrix());
  LocalMat& output = static_cast<LocalMat&>(input_grad);

  // Make dimensions and modes
  int dims = static_cast<int>(output_dims.size());
  auto input_modes = make_modes(dims);
  auto output_modes = make_modes(dims);
  for (int i = 0; i < dims; ++i) {
    if (input_dims[i] == 1 && input_dims[i] != output_dims[i]) {
      // Extra -1 skips mini batch dimension
      output_modes.erase(output_modes.begin() + (dims - i - 1));
    }
  }
  auto actual_output_dims = input_dims;
  for (int i = dims - 1; i >= 0; --i) {
    if (input_dims[i] == 1 && input_dims[i] != output_dims[i]) {
      actual_output_dims.erase(actual_output_dims.begin() + i);
    }
  }

  // Specify that the dimensions are given in row-major format
  RowMajorDims<int64_t> ct_input_dims(output_dims);
  RowMajorDims<int64_t> ct_output_dims(actual_output_dims);

  auto input_desc = get_descriptor(input, ct_input_dims);
  auto output_desc = get_descriptor(output, ct_output_dims);

  // Create workspace buffers
  LocalMat workspace;
  workspace.SetMemoryMode(1);
  auto handle = get_handle_ptr();
  uint64_t wspsize = 0;
  CHECK_CUTENSOR(
    cutensorReductionGetWorkspaceSize(handle,
                                      input.LockedBuffer(),
                                      &input_desc,
                                      input_modes.data(),
                                      output.LockedBuffer(),
                                      &output_desc,
                                      output_modes.data(),
                                      output.LockedBuffer(),
                                      &output_desc,
                                      output_modes.data(),
                                      CUTENSOR_OP_ADD,
                                      CUDATypeT<TensorDataType>::compute_type,
                                      &wspsize));
  workspace.Resize(wspsize, 1);

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(input_grad),
                                     gpu::get_sync_info(output_grad));

  // Compute reduction locally
  CHECK_CUTENSOR(cutensorReduction(
    handle,
    &one,
    input.LockedBuffer(),
    &input_desc,
    input_modes.data(),
    &zero,
    output.LockedBuffer(),
    &output_desc,
    output_modes.data(),
    output.Buffer(),
    &output_desc,
    output_modes.data(),
    CUTENSOR_OP_ADD,
    CUDATypeT<TensorDataType>::compute_type,
    workspace.Buffer(),
    wspsize,
    static_cast<El::SyncInfo<El::Device::GPU>>(multisync).Stream()));
}

} // namespace lbann

#endif // LBANN_HAS_CUTENSOR
