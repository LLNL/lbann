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

#include "lbann/comm.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/stack_trace.hpp"

#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/cudnn.hpp"
#endif

namespace lbann {

world_comm_ptr initialize(int& argc, char**& argv, int seed) {
  // Initialize Elemental.
  El::Initialize(argc, argv);
  // Create a new comm object.
  // Initial creation with every process in one model.
  auto comm = world_comm_ptr{new lbann_comm(0), &lbann::finalize };

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
    if (comm->get_rank_in_node() == 0) {
      std::cout << comm->get_rank_in_world() <<
                ": WARNING: node has " << num_numa_nodes <<
                " NUMA nodes but you have " << ppn << " processes per node" <<
                std::endl;
    }
  }
  hwloc_topology_destroy(topo);
#endif
  // Initialize local random number generators.
  init_random(seed);
  init_data_seq_random(seed);

  return comm;
}

void finalize(lbann_comm* comm) {
#ifdef LBANN_HAS_CUDNN
  cudnn::destroy();
#endif
  if (comm != nullptr) {
    delete comm;
  }
  El::Finalize();
}

/** hack to avoid long switch/case statement; users should ignore; of interest to developers */
static std::vector<std::string> pool_mode_names = { "invalid", "max", "average", "average_no_pad" };

/** returns a string representation of the pool_mode */
std::string get_pool_mode_name(pool_mode m) {
  if ((int)m < 1 or (int)m >= (int)pool_mode_names.size()) {
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: "
          + " Invalid pool_mode");
  }
  return pool_mode_names[(int)m];
}

} // namespace lbann
