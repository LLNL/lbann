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
//
// lbann_base .cpp - Basic definitions, functions
////////////////////////////////////////////////////////////////////////////////

#include "lbann/base.hpp"

#include <omp.h>
#if defined(LBANN_TOPO_AWARE)
#include <hwloc.h>
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif
#ifdef LBANN_HAS_SHMEM
#include <shmem.h>
#endif // LBANN_HAS_SHMEM

#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/stack_trace.hpp"

#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB
#ifdef LBANN_HAS_PYTHON
#include "lbann/utils/python.hpp"
#endif
#ifdef LBANN_HAS_NVSHMEM
#include "lbann/utils/nvshmem.hpp"
#endif
#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

#include <iostream>
#include <string>
#include <vector>

namespace lbann {
namespace {
lbann_comm* world_comm_ = nullptr;
}// namespace <anon>
namespace utils {
lbann_comm& get_current_comm() noexcept { return *world_comm_; }
}// namespace utils

MPI_Errhandler err_handle;

world_comm_ptr initialize(int& argc, char**& argv) {
  // Initialize Elemental.
  El::Initialize(argc, argv);

  // Create a new comm object.
  // Initial creation with every process in one model.
  auto comm = world_comm_ptr{new lbann_comm(0), &lbann::finalize };
  world_comm_ = comm.get();

  // Install MPI error handler
  MPI_Comm_create_errhandler(lbann_mpi_err_handler, &err_handle);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, err_handle);

#if defined(LBANN_TOPO_AWARE)
  // Determine the number of NUMA nodes present.
  hwloc_topology_t topo;
  hwloc_topology_init(&topo);
  hwloc_topology_load(topo);
  int numa_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_NUMANODE);
  if (numa_depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    std::cout << comm->get_rank_in_world() <<
              ": cannot determine hwloc NUMA-node depth" << std::endl;
  }
  int num_numa_nodes = hwloc_get_nbobjs_by_depth(topo, numa_depth);
  // Warn if there are more NUMA nodes than processes per node.
  // It's probably fine if there are more processes than NUMA nodes for now.
  // We can adjust that later when we better understand the threaded perf.
  int ppn = comm->get_procs_per_node();
  if (num_numa_nodes > ppn) {
    if (comm->get_rank_in_world() == 0) {
      std::cout << comm->get_rank_in_world() <<
                ": WARNING: node has " << num_numa_nodes <<
                " NUMA nodes but you have " << ppn << " processes per node" <<
                std::endl;
    }
  }
  hwloc_topology_destroy(topo);
#endif

#ifdef LBANN_HAS_SHMEM
  // Initialize SHMEM
  {
    int threading_level = SHMEM_THREAD_MULTIPLE;
    int status = shmem_init_thread(threading_level, &threading_level);
    if (status != 0 || threading_level != SHMEM_THREAD_MULTIPLE) {
      LBANN_ERROR("error initializing OpenSHMEM");
    }
  }
#endif // LBANN_HAS_SHMEM

#ifdef LBANN_HAS_DISTCONV
  dc::initialize(MPI_COMM_WORLD);
#endif // LBANN_HAS_DISTCONV

  return comm;
}

void finalize(lbann_comm* comm) {
#ifdef LBANN_HAS_NVSHMEM
  nvshmem::finalize();
#endif // LBANN_HAS_NVSHMEM
  MPI_Errhandler_free( &err_handle );
#ifdef LBANN_HAS_DISTCONV
  dc::finalize();
#endif
#ifdef LBANN_HAS_DNN_LIB
  dnn_lib::destroy();
#endif
#ifdef LBANN_HAS_PYTHON
  python::finalize();
#endif
#ifdef LBANN_HAS_NVSHMEM
  nvshmem::finalize();
#endif // LBANN_HAS_SHMEM
#ifdef LBANN_HAS_SHMEM
  shmem_finalize();
#endif // LBANN_HAS_SHMEM
  if (comm != nullptr) {
    delete comm;
  }
  El::Finalize();
}

matrix_format data_layout_to_matrix_format(data_layout layout) {
  matrix_format format;
  switch(layout) {
  case data_layout::MODEL_PARALLEL:
    format = matrix_format::MC_MR;
    break;
  case data_layout::DATA_PARALLEL:
    /// Weights are stored in STAR_STAR and data in STAR_VC
    format = matrix_format::STAR_STAR;
    break;
  default:
    LBANN_ERROR("Invalid data layout selected");
  }
  return format;
}

std::string to_string(data_layout const& dl) {
  switch (dl) {
  case data_layout::DATA_PARALLEL:
    return "data_parallel";
  case data_layout::MODEL_PARALLEL:
    return "model_parallel";
  case data_layout::invalid:
    return "invalid";
  }
  return "invalid data_layout";
}

data_layout data_layout_from_string(std::string const& str) {
  if (str == "data_parallel" || str == "DATA_PARALLEL")
    return data_layout::DATA_PARALLEL;
  if (str == "model_parallel" || str == "MODEL_PARALLEL")
    return data_layout::MODEL_PARALLEL;
  if (str == "invalid" || str == "INVALID")
    return data_layout::invalid; // Why is this a thing?
  LBANN_ERROR("Unable to convert \"", str, "\" to lbann::data_layout.");
}

std::string to_string(El::Device const& d) {
  switch (d) {
  case El::Device::CPU:
    return "CPU";
#ifdef HYDROGEN_HAVE_GPU
  case El::Device::GPU:
    return "GPU";
#endif // HYDROGEN_HAVE_GPU
  }
  return "invalid El::Device";
}

El::Device device_from_string(std::string const& str) {
  if (str == "cpu" || str == "CPU")
    return El::Device::CPU;
#ifdef HYDROGEN_HAVE_GPU
  if (str == "gpu" || str == "GPU")
    return El::Device::GPU;
#endif
  LBANN_ERROR("Unable to convert \"", str, "\" to El::Device.");
}

std::string to_string(execution_mode m) {
  switch(m) {
  case execution_mode::training:
    return "training";
  case execution_mode::validation:
    return "validation";
  case execution_mode::testing:
    return "testing";
  case execution_mode::prediction:
    return "prediction";
  case execution_mode::tournament:
    return "tournament";
  case execution_mode::invalid:
    return "invalid";
  default:
      LBANN_ERROR("Invalid execution mode specified");
  }
}

execution_mode exec_mode_from_string(std::string const& str) {
  if (str == "training" || str == "train")
    return execution_mode::training;
  else if (str == "validation" || str == "validate")
      return execution_mode::validation;
  else if (str == "testing" || str == "test")
    return execution_mode::testing;
  else if (str == "prediction" || str == "predict")
    return execution_mode::prediction;
  else if (str == "tournament")
    return execution_mode::tournament;
  else if (str == "invalid")
    return execution_mode::invalid;
  else
    LBANN_ERROR("\"" + str + "\" is not a valid execution mode.");
}

std::istream& operator>>(std::istream& is, execution_mode& m) {
  std::string tmp;
  is >> tmp;
  m = exec_mode_from_string(tmp);
  return is;
}

bool endsWith(const std::string mainStr, const std::string &toMatch)
{
  if(mainStr.size() >= toMatch.size() &&
     mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
    return true;
  else
    return false;
}

void print_matrix_dims(AbsDistMat *m, const char *name) {
  std::cout << "DISPLAY MATRIX: " << name << " = "
            << m->Height() << " x " << m->Width() << std::endl;
}

void print_local_matrix_dims(AbsMat *m, const char *name) {
  std::cout << "DISPLAY MATRIX: " << name << " = "
            << m->Height() << " x " << m->Width() << std::endl;
}

void lbann_mpi_err_handler(MPI_Comm *comm, int *err_code, ... ) {
  char err_string[MPI_MAX_ERROR_STRING];
  int err_string_length;
  MPI_Error_string(*err_code, &err_string[0], &err_string_length);
  LBANN_ERROR("MPI threw this error: ", err_string);
}

} // namespace lbann
