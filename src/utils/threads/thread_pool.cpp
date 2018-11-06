#include "lbann/utils/threads/thread_pool.hpp"

#include <algorithm>

namespace lbann {

thread_pool::thread_pool()
  : thread_joiner_{threads_},
    all_work_done_{false}
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


void thread_pool::do_thread_work_()
{
  while (not all_work_done_)
  {
    auto task = global_work_queue_.try_pop();
    if (task)
      (*task)();
    else
      std::this_thread::yield();
  }
}

}// namespace lbann
