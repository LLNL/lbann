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
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/threads/thread_topology.hpp"

#if defined(LBANN_TOPO_AWARE)
#include <hwloc.h>
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif

std::string print_hwloc_version(unsigned long ver) {
  return std::string{} + std::to_string(ver >> 16) + "." + std::to_string((ver & 0x00ff00) >> 8);
}

void check_hwloc_api_version() {
#if HWLOC_API_VERSION >= 0x00020000
  /* headers are recent */
  if (hwloc_get_api_version() < 0x20000) {
    LBANN_ERROR("HWLOC runtime library ", print_hwloc_version(hwloc_get_api_version()),
                " is older than 2.0 but LBANN was compile with HWLOC API version ",
                print_hwloc_version(HWLOC_API_VERSION));
  }
#else
  /* headers are pre-2.0 */
  if (hwloc_get_api_version() >= 0x20000) {
    LBANN_ERROR("HWLOC runtime library ", print_hwloc_version(hwloc_get_api_version()),
                " is more recent than 2.0 but LBANN was compile with HWLOC API version ",
                print_hwloc_version(HWLOC_API_VERSION));
  }
#endif
}
#endif

#include <algorithm>
#include <iostream>

namespace lbann {

thread_pool::thread_pool()
  : thread_joiner_{threads_},
    all_work_done_{false},
    m_threads_offset{0}
{
}

thread_pool::thread_pool(size_type max_threads)
  : thread_pool()
{
  size_type num_threads = std::max(max_threads,size_type{1});
  this->launch_threads(num_threads);
}

void thread_pool::launch_threads(size_type num_threads)
{
  threads_.reserve(num_threads);

  // Try to launch each worker thread
  try
  {
    for (size_type cnt = 0; cnt < num_threads; ++cnt) {
      threads_.emplace_back(&thread_pool::do_thread_work_,this);
    }
  }
  catch(...)
  {
    all_work_done_ = true;
    throw;
  }
}

// FIXME (trb 08/03/2019): Setting thread affinity is not a portable
// pthread operation (hence the _np suffix); indeed, OSX does not
// support it. Unfortunately the case on OSX is even more dire -- they
// seem to want to prevent you from messing with their scheduler at
// all. The MACH kernel API for doing this is marked "deprecated" and
// its use is not advised for code that is not tied to a specific OSX
// version (see here for more information:
// http://web.mit.edu/darwin/src/modules/xnu/osfmk/man/).
//
// As a result of the above, this will, in fact, *not* launch pinned
// threads when the locally-supported pthread API does not support it.
void thread_pool::launch_pinned_threads(
  size_type num_threads, int PU_offset) {

#if defined(LBANN_TOPO_AWARE)
  threads_.reserve(num_threads);
  m_work_group.reserve(num_threads);
  m_thread_id_to_local_id_map.reserve(num_threads);

  hwloc_topology_t topo;
  int err;
  /* initialize a topology context */
  err = hwloc_topology_init(&topo);
  if(err) { LBANN_ERROR("hwloc_topology_init failed"); }
  /* build the topology created and configured above */
  err = hwloc_topology_load(topo);
  if(err) { LBANN_ERROR("hwloc_topology_load failed"); }
  // Get the number of PUs per core
  // hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, 0);
  m_threads_offset = PU_offset;

  check_hwloc_api_version();
  //  hwloc_cpuset_t current_cpuset = get_local_cpuset_for_current_thread(topo);
  int i;
  //  hwloc_obj_t obj;
  // Try to launch each worker thread and pin it to a single core
  try
  {
    hwloc_cpuset_t allocated_cpuset = get_local_cpuset_for_current_thread(topo);
    hwloc_cpuset_t excluded_cpuset = hwloc_bitmap_alloc();
    // int cpuset_idx = 0;
    int skipped_indices = 0;
    // Skip PUs to match the thread offset
    hwloc_bitmap_foreach_begin(i, allocated_cpuset) {
      if(skipped_indices < m_threads_offset) {
        skipped_indices++;
        hwloc_bitmap_set(excluded_cpuset, i);
      }
    } hwloc_bitmap_foreach_end();

    hwloc_cpuset_t iot_cpuset = hwloc_bitmap_dup(allocated_cpuset);
    hwloc_bitmap_andnot(iot_cpuset, iot_cpuset, excluded_cpuset);
    if(hwloc_bitmap_iszero(iot_cpuset)) {
      LBANN_WARNING("Insufficient number of allocated cores to respect I/O CPU offset");
      hwloc_bitmap_free(iot_cpuset);
      // Reset the cpuset back to all of the allowed cores
      iot_cpuset = hwloc_bitmap_dup(allocated_cpuset);
    }

    for (size_type cnt = 0; cnt < num_threads; ++cnt) {
      hwloc_cpuset_t ht_cpuset = hwloc_bitmap_dup(iot_cpuset);
      hwloc_topology_t ht_topo;
      err = hwloc_topology_dup(&ht_topo, topo);
      if(err) { LBANN_ERROR("hwloc_topology_dup failed"); }
      threads_.emplace_back(&thread_pool::do_thread_work_pinned_thread_,
                            this, cnt, ht_topo, ht_cpuset);
    }
    hwloc_bitmap_free(iot_cpuset);
    hwloc_bitmap_free(excluded_cpuset);
    hwloc_bitmap_free(allocated_cpuset);
  }
  catch(...)
  {
    all_work_done_ = true;
    throw;
  }
#else
  launch_threads(num_threads);
#endif// LBANN_HAS_PTHREAD_AFFINITY_SUPPORT
}

void thread_pool::reap_threads() {
  if (this->get_num_threads() == 0) {
    return;
  }
  all_work_done_ = true;
  do {
    global_work_queue_.wake_all(true);
  }while(!global_work_queue_.empty());

  for (auto& t : threads_) if (t.joinable()) t.join();

  m_work_group.clear();
  m_thread_id_to_local_id_map.clear();
  threads_.clear();
  /// Reset the flag so that new threads can be started
  all_work_done_ = false;
  global_work_queue_.set_stop_threads(false);
  return;
}

void thread_pool::relaunch_pinned_threads(size_type num_threads) {
  reap_threads();
  launch_pinned_threads(num_threads, m_threads_offset);
  return;
}

void thread_pool::do_thread_work_()
{
  while (not all_work_done_)
  {
    auto task = global_work_queue_.wait_and_pop();
    if (task) {
      (*task)();
    }
  }
}

#if defined(LBANN_TOPO_AWARE)
void thread_pool::do_thread_work_pinned_thread_(int tid, hwloc_topology_t topo, hwloc_cpuset_t cpuset)
{
  // Set the CPU affinity for the thread
  auto error = hwloc_set_cpubind(topo, cpuset, 0);
  // Free the hwloc_cpuset_t structure once the thread is pinned
  hwloc_bitmap_free(cpuset);
  //assert(!err);
  if (error != 0) {
    std::cerr << "error in hwloc_set_cpubind, error="
              << strerror(error) << std::endl;
  }

  /* terminate this topology context */
  hwloc_topology_destroy(topo);

  {
    std::lock_guard<std::mutex> guard(m_thread_map_mutex);
    // Establish a local thread id
    std::thread::id this_id = std::this_thread::get_id();
    m_thread_id_to_local_id_map[this_id] = tid;
  }
  while (not all_work_done_)
  {
    auto task = global_work_queue_.wait_and_pop();
    if (task) {
      (*task)();
    }
  }
}
#endif // LBANN_TOPO_AWARE

int thread_pool::get_local_thread_id() {
  std::thread::id this_id = std::this_thread::get_id();
  return m_thread_id_to_local_id_map[this_id];
}

}// namespace lbann
