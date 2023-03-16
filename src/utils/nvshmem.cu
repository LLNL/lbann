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

#include "lbann/utils/nvshmem.hpp"
#ifdef LBANN_HAS_NVSHMEM

namespace lbann {
namespace nvshmem {

namespace {

bool is_initialized_ = false;
bool is_finalized_ = false;

} // namespace

bool is_initialized() noexcept { return is_initialized_; }

bool is_finalized() noexcept { return is_finalized_; }

bool is_active() noexcept { return is_initialized() && !is_finalized(); }

void initialize(MPI_Comm comm)
{

  // Check if NVSHMEM has already been initialized or finalized
  if (is_active()) {
    return;
  }
  if (is_finalized()) {
    LBANN_ERROR("attempted to initialize NVSHMEM after it has been finalized");
  }

  // Initialize NVSHMEM
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  auto status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  if (status != 0) {
    LBANN_ERROR("failed to initialize NVSHMEM (status ", status, ")");
  }
  is_initialized_ = true;
}

void finalize()
{
  if (is_active()) {
    nvshmem_finalize();
    is_finalized_ = true;
  }
}

} // namespace nvshmem
} // namespace lbann

#endif // LBANN_HAS_NVSHMEM
