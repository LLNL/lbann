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

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

lbann_comm* initialize(int& argc, char**& argv, int seed) {
  // Initialize Elemental.
  El::Initialize(argc, argv);
  // Create a new comm object.
  // Initial creation with every process in one model.
  lbann_comm* comm = new lbann_comm(0);
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
  // Initialize local random number generators.
  init_random(seed);
  init_data_seq_random(seed);
  return comm;
}

void finalize(lbann_comm* comm) {
  if (comm != nullptr) {
    delete comm;
  }
  El::Finalize();
}

}  // namespace lbann

/** hack to avoid long switch/case statement; users should ignore; of interest to developers */
static std::vector<std::string> weight_initialization_names  = 
    { "zero", "uniform", "normal", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform"};

/** returns a string representation of the weight_initialization */
std::string get_weight_initialization_name(weight_initialization m) {
  if ((int)m < 0 or (int)m >= (int)weight_initialization_names.size()) {
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: "
          + " Invalid weight_initialization");
  }
  return weight_initialization_names[(int)m];
}

/** hack to avoid long switch/case statement; users should ignore; of interest to developers */
static std::vector<std::string> pool_mode_names = { "max", "average", "average_no_pad" };

/** returns a string representation of the pool_mode */
std::string get_pool_mode_name(pool_mode m) {
  if ((int)m < 0 or (int)m >= (int)pool_mode_names.size()) {
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: "
          + " Invalid pool_mode");
  }
  return pool_mode_names[(int)m];
}

