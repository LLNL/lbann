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

#include "lbann/utils/system_info.hpp"

#include "lbann/utils/environment_variable.hpp"

#include <stdexcept>
#include <string>

#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>

namespace lbann {
namespace utils {
namespace {

int try_mpi_comm_rank() noexcept
{
  int rank = -1;
  int mpi_has_been_initialized = -1, mpi_has_been_finalized = -1;
  MPI_Initialized(&mpi_has_been_initialized);
  MPI_Finalized(&mpi_has_been_finalized);

  if (mpi_has_been_initialized && !mpi_has_been_finalized)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  return rank;
}

// I know about SLURM, Open-MPI, and MVAPICH2
int try_env_variable_rank()
{
  ENV slurm_rank("SLURM_PROCID");
  if (slurm_rank.exists())
    return slurm_rank.template value<int>();
  ENV openmpi_rank("OMPI_COMM_WORLD_RANK");
  if (openmpi_rank.exists())
    return openmpi_rank.template value<int>();
  ENV mvapich2_rank("MV2_COMM_WORLD_RANK");
  if (mvapich2_rank.exists())
    return mvapich2_rank.template value<int>();

  return -1;
}

int try_mpi_comm_size() noexcept
{
  int size = -1;
  int mpi_has_been_initialized = -1, mpi_has_been_finalized = -1;
  MPI_Initialized(&mpi_has_been_initialized);
  MPI_Finalized(&mpi_has_been_finalized);

  if (mpi_has_been_initialized && !mpi_has_been_finalized)
    MPI_Comm_size(MPI_COMM_WORLD, &size);

  return size;
}

// I know about SLURM, Open-MPI, and MVAPICH2
int try_env_variable_size()
{
  ENV slurm_size("SLURM_NTASKS");
  if (slurm_size.exists())
    return slurm_size.template value<int>();
  ENV openmpi_size("OMPI_COMM_WORLD_SIZE");
  if (openmpi_size.exists())
    return openmpi_size.template value<int>();
  ENV mvapich2_size("MV2_COMM_WORLD_SIZE");
  if (mvapich2_size.exists())
    return mvapich2_size.template value<int>();

  return -1;
}
}// namespace <anon>

std::string SystemInfo::pid() const
{
  return std::to_string(getpid());
}

std::string SystemInfo::host_name() const
{
  char hostname[4096];
  int status = gethostname(hostname, 4096);
  if (status != 0)
    throw std::runtime_error("gethostname failed");
  return hostname;
}

int SystemInfo::mpi_rank() const
{
  static int rank = -1;

  // Short-circuit if rank has already been found.
  if (rank != -1)
    return rank;

  // First try MPI directly
  rank = try_mpi_comm_rank();

  // Now try some environment variables
  if (rank == -1)
    rank = try_env_variable_rank();

  // At this point, I assume I'm not in an MPI job.
  if (rank == -1)
    rank = 0;

  return rank;
}

int SystemInfo::mpi_size() const
{
  static int size = -1;

  // Short-circuit if size has already been found.
  if (size != -1)
    return size;

  // First try MPI directly
  size = try_mpi_comm_size();

  // Now try some environment variables
  if (size == -1)
    size = try_env_variable_size();

  // At this point, I assume I'm not in an MPI job.
  if (size == -1)
    size = 0;

  return size;
}

std::string
SystemInfo::env_variable_value(std::string const& var_name) const
{
  return ENV(var_name).raw_value();
}

}// namespace utils
}// namespace lbann
