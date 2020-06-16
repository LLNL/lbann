#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/random_number_generators.hpp"

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
  size_type num_threads, int cpu_offset) {
#ifdef LBANN_HAS_PTHREAD_AFFINITY_SUPPORT
  threads_.reserve(num_threads);
  m_work_group.reserve(num_threads);
  m_thread_id_to_local_id_map.reserve(num_threads);

  m_threads_offset = cpu_offset;

  // Find the current thread affinity
  cpu_set_t cpuset, ht_cpuset;
  CPU_ZERO(&cpuset);

  auto error = pthread_getaffinity_np(pthread_self(),
                                      sizeof(cpu_set_t), &cpuset);
  if (error != 0) {
    std::cerr << "error in pthread_getaffinity_np, error=" << error
              << std::endl;
  }

  // Try to launch each worker thread
  try
  {
    for (size_type cnt = 0; cnt < num_threads; ++cnt) {
      CPU_ZERO(&ht_cpuset);
      // Pin this thread to the base CPU id plus the thread count and offset
      for (int j = 0; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &cpuset)) {
          CPU_SET(j+cnt+cpu_offset, &ht_cpuset);
        }
      }

      threads_.emplace_back(&thread_pool::do_thread_work_pinned_thread_,
                            this, cnt, ht_cpuset);
    }
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

#ifdef LBANN_HAS_PTHREAD_AFFINITY_SUPPORT
void thread_pool::do_thread_work_pinned_thread_(int tid, cpu_set_t cpu_set)
{
  // Set the CPU affinity for the thread
  auto error = pthread_setaffinity_np(pthread_self(),
                                      sizeof(cpu_set_t), &cpu_set);
  if (error != 0) {
    std::cerr << "error in pthread_setaffinity_np, error="
              << error << std::endl;
  }

  {
    std::lock_guard<std::mutex> guard(m_thread_map_mutex);
    // Establish a local thread id
    std::thread::id this_id = std::this_thread::get_id();
    m_thread_id_to_local_id_map[this_id] = tid;
  }
  init_io_generator(tid);
  init_fast_io_generator(tid);

  while (not all_work_done_)
  {
    auto task = global_work_queue_.wait_and_pop();
    if (task) {
      (*task)();
    }
  }
}
#endif // LBANN_HAS_PTHREAD_AFFINITY_SUPPORT

int thread_pool::get_local_thread_id() {
  std::thread::id this_id = std::this_thread::get_id();
  return m_thread_id_to_local_id_map[this_id];
}

}// namespace lbann
