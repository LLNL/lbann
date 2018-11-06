#ifndef __LBANN_THREAD_POOL_HPP__
#define __LBANN_THREAD_POOL_HPP__

#include <future>
#include <thread>
#include <vector>

#include "thread_safe_queue.hpp"
#include "type_erased_function.hpp"

namespace lbann {

class thread_pool {
public:
  using thread_container_type = std::vector<std::thread>;
  using size_type = typename thread_container_type::size_type;

private:
  /** \class thread_joiner
   *  \brief RAII object that destroys threads
   */
  struct thread_joiner
  {
    /** \brief Grab container reference */
    thread_joiner(thread_container_type& threads) : threads_(threads) {}
    /** \brief Destructor: safely shut all threads down */
    ~thread_joiner() { for (auto& t : threads_) if (t.joinable()) t.join(); }
    /** \brief Thread container reference */
    thread_container_type& threads_;
  };

public:
  /** \brief Construct an empty threadpool. Size must be set with launch().
   */
  thread_pool();

  /** \brief Construct a threadpool of a given size.
   *
   *  \param max_threads Total threads available. max_threads-1 worker
   *                     threads will be launched.
   */
  thread_pool(size_type max_threads);

  /** \brief Destroy the threadpool */
  ~thread_pool() { all_work_done_ = true; }

  /** \brief Launch the threads */
  void launch_threads(size_type num_threads);

  /** \brief Submit a job to the pool's queue */
  template <typename FunctionT>
  std::future<typename std::result_of<FunctionT()>::type>
  submit_job(FunctionT func)
  {
    using return_type = typename std::result_of<FunctionT()>::type;

    std::packaged_task<return_type()> task(std::move(func));
    auto future = task.get_future();
    global_work_queue_.push(std::move(task));
    return future;
  }

  /** Query the number of worker threads actually present */
  size_type get_num_threads() const noexcept { return threads_.size(); }

private:
  /** \brief The task executed by each thread */
  void do_thread_work_();

private:

  /** \brief Container holding the threads */
  thread_container_type threads_;

  /** \brief The thread-safe work queue */
  thread_safe_queue<type_erased_function> global_work_queue_;

  /** \brief RAII "deleter" for the threads */
  thread_joiner thread_joiner_;

  /** \brief Flag to track if more work is to be done */
  std::atomic<bool> all_work_done_;

};// class thread_pool

}// namespace lbann
#endif /* __LBANN_THREAD_POOL_HPP__ */
