////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include <lbann/utils/exception.hpp>
#include <lbann/utils/threads/thread_topology.hpp>

#if defined(LBANN_TOPO_AWARE)
#include <hwloc.h>
#ifdef LBANN_HAS_CUDA
#include <hwloc/cudart.h>
#endif // LBANN_HAS_CUDA
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#ifdef LBANN_HAS_CUDA
#include <hydrogen/device/gpu/CUDA.hpp>
#endif // LBANN_HAS_CUDA

#include <iostream>

namespace lbann {

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
    printf("%u cores\n", hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE));
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
#if HWLOC_API_VERSION >= 0x00020000
    while (parent && !parent->attr->numanode.local_memory)
      parent = parent->parent;
    printf("%llu bytes\n",
           (unsigned long long)parent->attr->numanode.local_memory);
#else
    while (parent && !parent->memory.local_memory)
      parent = parent->parent;
    printf("%llu bytes\n", (unsigned long long)parent->memory.local_memory);
#endif
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
// Used by thread_pool.hpp also -- NOT static.
hwloc_cpuset_t get_local_cpuset_for_current_thread(hwloc_topology_t topo)
{
  hwloc_cpuset_t local_cpuset = hwloc_bitmap_alloc();
#ifdef LBANN_HAS_CUDA
  // Find CPUs close to the GPU being used
  hwloc_cudart_get_device_cpuset(topo,
                                 hydrogen::gpu::DefaultDevice(),
                                 local_cpuset);
#else
  hwloc_const_cpuset_t allowed_cpuset = hwloc_topology_get_allowed_cpuset(topo);
  hwloc_bitmap_free(local_cpuset);
  local_cpuset = hwloc_bitmap_dup(allowed_cpuset);
  //  hwloc_bitmap_free(allowed_cpuset);
#endif // LBANN_HAS_CUDA
  return local_cpuset;
}

#endif // LBANN_TOPO_AWARE

} // namespace lbann
