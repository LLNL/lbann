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

#include "lbann/utils/cudnn_wrapper.hpp"
#include "El.hpp"

namespace lbann {
namespace cudnn {

/// @todo Efficient implementation
void cudnn_manager::global_allreduce_on_gpus(GPUMat& data,
                                             El::mpi::Comm comm) {
  if (is_nccl_used()) {
#ifdef LBANN_HAS_NCCL2
    /// @todo What if m_nccl_comm doesn't match comm?
    global_allreduce_on_gpus_nccl(data);
    synchronize();
#else
    LBANN_ERROR("NCCL not detected");
#endif // LBANN_HAS_NCCL2
  } else {
    static CPUMat workspace;
    El::Copy(data, workspace);
    El::AllReduce(workspace, comm);
    El::Copy(workspace, data);
  }
}

#ifdef LBANN_HAS_NCCL2
namespace {

/** Get NCCL datatype corresponding to lbann::DataType. */
inline ncclDataType_t get_nccl_datatype() {
  switch (sizeof(DataType)) {
    case 8:  return ncclDouble;
    case 4:  return ncclFloat;
    case 2:  return ncclHalf;
    default: LBANN_ERROR("invalid data type for NCCL");
  }
  return ncclFloat;
}

} // namespace

void cudnn_manager::global_allreduce_on_gpus_nccl(GPUMat& data) {
  const El::Int size = data.Height() * data.Width();
  if (data.Width() <= 1 || data.Height() == data.LDim()) {
    // Perform allreduce on contiguous data
    NCCLCHECK(ncclAllReduce(data.LockedBuffer(),
                            data.Buffer(),
                            size,
                            get_nccl_datatype(),
                            ncclSum,
                            m_nccl_comm[0],
                            get_stream()));
  } else {
    // Copy non-contiguous data into contiguous workspace before
    // performing allreduce
    GPUMat workspace;
#ifdef HYDROGEN_HAVE_CUB
    workspace.SetMemoryMode(1); // CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
    if (size * sizeof(DataType) > get_workspace_size()) {
      LBANN_ERROR("insufficient GPU workspace");
    }
    El::Copy(data, workspace);
    NCCLCHECK(ncclAllReduce(workspace.LockedBuffer(),
                            workspace.Buffer(),
                            size,
                            get_nccl_datatype(),
                            ncclSum,
                            m_nccl_comm[0],
                            get_stream()));
    El::Copy(workspace, data);
  }
}
#endif // LBANN_HAS_NCCL2

} // namespace cudnn
} // namespace lbann
