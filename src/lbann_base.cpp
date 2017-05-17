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
//
// lbann_base .cpp - Basic definitions, functions
////////////////////////////////////////////////////////////////////////////////

#include <thread>
#include <omp.h>
#include <hwloc.h>
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
  #define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

namespace lbann {

void initialize(lbann_comm* comm) {
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
#ifdef _OPENMP
  // Initialize the default number of threads to use for parallel regions.
  // Note the num_threads directive can override this if specifically set.
  // Further, if the OMP_NUM_THREADS environment variable is set, we don't
  // change it.
  if (getenv("OMP_NUM_THREADS") == NULL) {
    const int threads_per_rank = std::thread::hardware_concurrency() / ppn;
    omp_set_num_threads(threads_per_rank);
  }
#endif  // _OPENMP
}

void finalize() {

}

}  // namespace lbann
