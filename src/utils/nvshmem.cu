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

#include "lbann/utils/nvshmem.hpp"
#ifdef LBANN_HAS_NVSHMEM
#include "lbann/utils/exception.hpp"

namespace lbann {
namespace nvshmem {

void initialize(MPI_Comm comm) {
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  auto status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  if (status != 0) {
    LBANN_ERROR("failed to initialize NVSHMEM (status ",status,")");
  }
}

void finalize() {
  nvshmem_finalize();
}

} // namespace nvshmem
} // namespace lbann

#endif // LBANN_HAS_NVSHMEM
