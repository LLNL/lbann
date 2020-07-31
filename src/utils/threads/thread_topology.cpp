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

#include <lbann/utils/threads/thread_topology.hpp>
#include <lbann/utils/exception.hpp>

#if defined(LBANN_TOPO_AWARE)
#include <hwloc.h>
#ifdef LBANN_HAS_GPU
#include <hwloc/cudart.h>
#endif // LBANN_HAS_GPU
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#ifdef LBANN_HAS_GPU
#include <hydrogen/device/gpu/CUDA.hpp>
#endif // LBANN_HAS_GPU

#include <iostream>

namespace lbann {

int get_num_numa_nodes() {
  int num_numa_nodes = 1;
#if defined(LBANN_TOPO_AWARE)
  // Determine the number of NUMA nodes present.
  hwloc_topology_t topo;
  hwloc_topology_init(&topo);
  hwloc_topology_load(topo);
  int numa_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_NUMANODE);
  // if (numa_depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
  //   LBANN_ERROR(comm->get_rank_in_world(),
  //               ": cannot determine hwloc NUMA-node depth");
  // }
  num_numa_nodes = hwloc_get_nbobjs_by_depth(topo, numa_depth);
  // Warn if there are more NUMA nodes than processes per node.
  // It's probably fine if there are more processes than NUMA nodes for now.
  // We can adjust that later when we better understand the threaded perf.
  // ppn = comm->get_procs_per_node();
  // if (num_numa_nodes > ppn) {
  //   // if (comm->get_rank_in_node() == 0) {
  //     std::cout << comm->get_rank_in_world() <<
  //               ": WARNING: node has " << num_numa_nodes <<
  //               " NUMA nodes but you have " << ppn << " processes per node" <<
  //               std::endl;
  //   // }
  // }
  hwloc_topology_destroy(topo);
#endif // LBANN_TOPO_AWARE
  return num_numa_nodes;
}

#if defined(LBANN_TOPO_AWARE)
void hwloc_print_topo()
{
  hwloc_topology_t topo;
  int err;
  /* initialize a topology context */
  err = hwloc_topology_init(&topo);
  assert(!err);
  /* build the topology created and configured above */
  err = hwloc_topology_load(topo);

  {
    printf("%u cores\n",
	   hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE));
  }

  {
    hwloc_obj_t core3, core7;
    core3 = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, 3);
    core7 = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, 7);
    if (core3 && core7) {
      hwloc_obj_t ancestor = hwloc_get_common_ancestor_obj(topo, core3, core7);
      printf("ancestor type %s\n", hwloc_obj_type_string(ancestor->type));
    }
  }

  {
    hwloc_obj_t core0 = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, 0);
    hwloc_obj_t parent = core0;
    while (parent && !parent->memory.local_memory)
      parent = parent->parent;
    printf("%llu bytes\n", (unsigned long long) parent->memory.local_memory);
  }

  {
    hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, 0);
    while (pu) {
      printf("%u\n", pu->os_index);
      pu = pu->next_cousin;
    }
  }

  /* terminate this topology context */
  hwloc_topology_destroy(topo);
  std::cout << err << std::endl;
  return;
}

// This function is implemented in HWLOC 2.1
int hwloc_bitmap_singlify_per_core(hwloc_topology_t topology, hwloc_bitmap_t cpuset, unsigned which)
{
  hwloc_obj_t core = NULL;
  while ((core = hwloc_get_next_obj_covering_cpuset_by_type(topology, cpuset, HWLOC_OBJ_CORE, core)) != NULL) {
    /* this core has some PUs in the cpuset, find the index-th one */
    unsigned i = 0;
    int pu = -1;
    do {
      pu = hwloc_bitmap_next(core->cpuset, pu);
      if (pu == -1) {
        /* no which-th PU in cpuset and core, remove the entire core */
        hwloc_bitmap_andnot(cpuset, cpuset, core->cpuset);
        break;
      }
      if (hwloc_bitmap_isset(cpuset, pu)) {
        if (i == which) {
          /* remove the entire core except that exact pu */
          hwloc_bitmap_andnot(cpuset, cpuset, core->cpuset);
          hwloc_bitmap_set(cpuset, pu);
          break;
        }
        i++;
      }
    } while (1);
  }
  return 0;
}

hwloc_cpuset_t get_local_cpuset_for_current_thread(hwloc_topology_t topo) {
  hwloc_cpuset_t local_cpuset = hwloc_bitmap_alloc();
#ifdef LBANN_HAS_GPU
  // Find CPUs close to the GPU being used
  hwloc_cudart_get_device_cpuset(topo, hydrogen::gpu::DefaultDevice(), local_cpuset);
#else
  hwloc_const_cpuset_t allowed_cpuset = hwloc_topology_get_allowed_cpuset(topo);
  local_cpuset = hwloc_bitmap_dup(allowed_cpuset);
  //  hwloc_bitmap_free(allowed_cpuset);
#endif // LBANN_HAS_GPU
  return local_cpuset;
}

hwloc_cpuset_t get_allocated_cpuset_for_current_thread(hwloc_topology_t topo) {
  hwloc_cpuset_t current_cpuset = hwloc_bitmap_alloc();
  int err = hwloc_get_cpubind(topo, current_cpuset, 0);
  if(err) { LBANN_ERROR("Unable to execute hwloc_get_cpubind"); }

  hwloc_cpuset_t PU_core_set = hwloc_bitmap_alloc();
  // Primary core set
  hwloc_cpuset_t primary_PU_core_set = hwloc_bitmap_dup(current_cpuset);
  // Hyperthread core set
  hwloc_cpuset_t ht_PU_core_set = hwloc_bitmap_dup(current_cpuset);
  // Get the list of available cores without hyperthreads
  err = hwloc_bitmap_singlify_per_core(topo, primary_PU_core_set, 0);
  if(err) { LBANN_ERROR("Unable to singlify the cpuset"); }
  err = hwloc_bitmap_singlify_per_core(topo, ht_PU_core_set, 1);
  if(err) { LBANN_ERROR("Unable to singlify the cpuset"); }

  if(!hwloc_bitmap_iszero(ht_PU_core_set)) {
    find_common_core_set_from_cpu_masks(topo, PU_core_set, primary_PU_core_set, ht_PU_core_set);
  }else {
    PU_core_set = hwloc_bitmap_dup(primary_PU_core_set);
  }

  hwloc_bitmap_free(current_cpuset);
  hwloc_bitmap_free(primary_PU_core_set);
  hwloc_bitmap_free(ht_PU_core_set);

  return PU_core_set;
}


void find_common_core_set_from_cpu_masks(hwloc_topology_t topo,
                                         hwloc_bitmap_t core_set,
                                         hwloc_bitmap_t primary_set,
                                         hwloc_bitmap_t ht_set) {
  hwloc_cpuset_t tmp_primary_set = hwloc_bitmap_alloc();
  hwloc_cpuset_t tmp_ht_set = hwloc_bitmap_alloc();
  // Find the set of cores in the mask of the primary set
  {
    hwloc_obj_t core = NULL;
    while ((core = hwloc_get_next_obj_covering_cpuset_by_type(topo, primary_set, HWLOC_OBJ_CORE, core)) != NULL) {
      hwloc_bitmap_or(tmp_primary_set, tmp_primary_set, core->cpuset);
    }
  }
  // Find the set of cores in the mask for the secondary (hyperthread) set
  {
    hwloc_obj_t core = NULL;
    while ((core = hwloc_get_next_obj_covering_cpuset_by_type(topo, ht_set, HWLOC_OBJ_CORE, core)) != NULL) {
      hwloc_bitmap_or(tmp_ht_set, tmp_ht_set, core->cpuset);
    }
  }
  // AND both sets together to find the actual cores in the CPU mask
  hwloc_bitmap_and(core_set, tmp_primary_set, tmp_ht_set);

  hwloc_bitmap_free(tmp_primary_set);
  hwloc_bitmap_free(tmp_ht_set);
}
#endif // LBANN_TOPO_AWARE

} // namespace lbann
