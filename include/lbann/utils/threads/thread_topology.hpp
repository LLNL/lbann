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

#ifndef LBANN_UTILS_HW_TOPOLOGY_HPP_INCLUDED
#define LBANN_UTILS_HW_TOPOLOGY_HPP_INCLUDED

// Defines, among other things, DataType.
#include "lbann_config.hpp"

#if defined(LBANN_TOPO_AWARE)
#include <hwloc.h>
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

namespace lbann {
  int get_num_numa_nodes();
#if defined(LBANN_TOPO_AWARE)
  void hwloc_print_topo();

#if HWLOC_API_VERSION < 0x00020000
  // This function is implemented in HWLOC 2.1
  int hwloc_bitmap_singlify_per_core(hwloc_topology_t topology, hwloc_bitmap_t cpuset, unsigned which);
#endif

  hwloc_cpuset_t get_local_cpuset_for_current_thread(hwloc_topology_t topo);

  /** @brief Return the allocated cpuset for the current thread, masking out
   *  PUs from spurious "unbound" cores.
   *  Allocates a bitmap that must be freed by calling function */
  hwloc_cpuset_t get_allocated_cpuset_for_current_thread(const hwloc_topology_t topo);

  /** @brief Given two sets of CPU bitmaps return the common set of
      cores */
  void find_common_core_set_from_cpu_masks(hwloc_topology_t topo,
                                           hwloc_bitmap_t core_set,
                                           hwloc_bitmap_t primary_set,
                                           hwloc_bitmap_t ht_set);
#endif // LBANN_TOPO_AWAR
}

#endif // LBANN_UTILS_HW_TOPOLOGY_HPP_INCLUDED
