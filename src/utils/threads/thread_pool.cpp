#include "lbann/utils/threads/thread_pool.hpp"

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
      // pthread_t my_thread_native = threads_[cnt].native_handle();
      // cpu_set_t io_cpu_mask;
      // // CPU_ZERO(&cpu_mask);
      // // sched_getaffinity(0, sizeof(cpu_mask), &cpu_mask);
      // //    if ( (rc = sched_getaffinity(0, sizeof(cpu_mask), &cpu_mask)) < 0 ) {
      // //      perror("sched_getaffinity");
      // //      return rc;
      // //    }
      // CPU_ZERO(&io_cpu_mask);
      // CPU_SET(44, &io_cpu_mask);
      // //    int rc = sched_setaffinity(0, sizeof(cpu_set_t), &io_cpu_mask);
      // int rc = pthread_setaffinity_np(my_thread_native, sizeof(cpu_set_t), &io_cpu_mask);
      // if(rc < 0) {
      //   perror("sched_set_affinity");
      // }

    }
  }
  catch(...)
  {
    all_work_done_ = true;
    throw;
  }
}

  void thread_pool::launch_pinned_threads(size_type num_threads, int cpu_offset)
{
  threads_.reserve(num_threads);
  m_work_group.reserve(num_threads);

  m_threads_offset = cpu_offset;

  // Find the current thread affinity
  cpu_set_t cpuset, ht_cpuset;
  CPU_ZERO(&cpuset);

  auto error = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
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

      threads_.emplace_back(&thread_pool::do_thread_work_pinned_thread_,this, cnt, ht_cpuset);
      // pthread_t my_thread_native = threads_[cnt].native_handle();
      // cpu_set_t io_cpu_mask;
      // // CPU_ZERO(&cpu_mask);
      // // sched_getaffinity(0, sizeof(cpu_mask), &cpu_mask);
      // //    if ( (rc = sched_getaffinity(0, sizeof(cpu_mask), &cpu_mask)) < 0 ) {
      // //      perror("sched_getaffinity");
      // //      return rc;
      // //    }
      // CPU_ZERO(&io_cpu_mask);
      // CPU_SET(44, &io_cpu_mask);
      // //    int rc = sched_setaffinity(0, sizeof(cpu_set_t), &io_cpu_mask);
      // int rc = pthread_setaffinity_np(my_thread_native, sizeof(cpu_set_t), &io_cpu_mask);
      // if(rc < 0) {
      //   perror("sched_set_affinity");
      // }

    }
  }
  catch(...)
  {
    all_work_done_ = true;
    throw;
  }
}

void thread_pool::reap_threads() {
  std::cout << "About to reap the I/O threasd" << std::endl;
  all_work_done_ = true;
  while(!global_work_queue_.empty()) { global_work_queue_.wake_all(true); }

  m_work_group.clear();
  m_thread_id_to_local_id_map.clear();
  threads_.clear();
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

void thread_pool::do_thread_work_pinned_thread_(int tid, cpu_set_t cpu_set)
{
  // Set the CPU affinity for the thread
  auto error = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);
  if (error != 0) {
    std::cerr << "error in pthread_setaffinity_np, error=" << error << std::endl;
  }

  // std::cout  << "I have pined thread " << tid << " to {";

  // for (int j = 0; j < CPU_SETSIZE; j++) {
  //   if (CPU_ISSET(j, &cpu_set)) {
  //     std::cout << " " << j;
  //   }
  // }
  // std::cout << " }" << std::endl;
  {
    std::lock_guard<std::mutex> guard(m_thread_map_mutex);
    // Establish a local thread id
    std::thread::id this_id = std::this_thread::get_id();
    m_thread_id_to_local_id_map[this_id] = tid;

    // Iterate and print keys and values of unordered_map
    // for( const auto& n : m_thread_id_to_local_id_map ) {
    // std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
    // }
  }
  while (not all_work_done_)
  {
    auto task = global_work_queue_.wait_and_pop();
    if (task) {
      (*task)();
    }
  }
}

int thread_pool::get_local_thread_id() {
  std::thread::id this_id = std::this_thread::get_id();
  return m_thread_id_to_local_id_map[this_id];
}

}// namespace lbann
