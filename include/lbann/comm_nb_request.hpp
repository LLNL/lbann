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

#ifndef LBANN_COMM_NB_REQUEST_HPP_INCLUDED
#define LBANN_COMM_NB_REQUEST_HPP_INCLUDED

#include "lbann_config.hpp"

#ifdef LBANN_HAS_ALUMINUM
#include <Al.hpp>
#endif // LBANN_HAS_ALUMINUM

namespace lbann {

namespace Al {

/** Dummy Aluminum backend. */
class dummy_backend
{
public:
  using req_type = int;
  static constexpr req_type null_req = 0;
};

// Define aliases for Aluminum backends
#ifdef LBANN_HAS_ALUMINUM
using mpi_backend = ::Al::MPIBackend;
#else
using mpi_backend = lbann::Al::dummy_backend;
#endif // LBANN_HAS_ALUMINUM
using mpi_req_type = mpi_backend::req_type;
static const mpi_req_type mpi_null_req = mpi_backend::null_req;
/// @todo MPI-CUDA backend
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)
using nccl_backend = ::Al::NCCLBackend;
// LBANN does its own synchronization on this.
#else
using nccl_backend = lbann::Al::dummy_backend;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)
using nccl_req_type = nccl_backend::req_type;
static const nccl_req_type nccl_null_req = nccl_backend::null_req;
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_HOST_TRANSFER)
using hosttransfer_backend = ::Al::HostTransferBackend;
#else
using hosttransfer_backend = lbann::Al::dummy_backend;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_HOST_TRANSFER)
using hosttransfer_req_type = hosttransfer_backend::req_type;
static const hosttransfer_req_type hosttransfer_null_req =
  hosttransfer_backend::null_req;

/** Wrapper for Aluminum non-blocking routine requests. */
struct request
{
  mpi_req_type mpi_req = mpi_null_req;
  nccl_req_type nccl_req = nccl_null_req;
  hosttransfer_req_type hosttransfer_req = hosttransfer_null_req;
  MPI_Request raw_mpi_req = MPI_REQUEST_NULL;
};
} // namespace Al

} // namespace lbann

#endif // LBANN_COMM_NB_REQUEST_HPP_INCLUDED
