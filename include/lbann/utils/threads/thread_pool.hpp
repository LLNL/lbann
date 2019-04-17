#ifndef __LBANN_THREAD_POOL_HPP__
#define __LBANN_THREAD_POOL_HPP__

#include <future>
#include <thread>
#include <vector>
#include <unordered_map>

#include "thread_safe_queue.hpp"
#include "type_erased_function.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

class thread_pool {
public:
  using thread_container_type = std::vector<std::thread>;
  using size_type = typename thread_container_type::size_type;

private:
  /** @class thread_joiner
   *  @brief RAII object that destroys threads
   */
  struct thread_joiner
  {
    /** @brief Grab container reference */
    thread_joiner(thread_container_type& threads) : threads_(threads) {}
    /** @brief Destructor: safely shut all threads down */
    ~thread_joiner() { for (auto& t : threads_) if (t.joinable()) t.join(); }
    /** @brief Thread container reference */
    thread_container_type& threads_;
  };

public:
  /** @brief Construct an empty threadpool. Size must be set with launch().
   */
  thread_pool();

  /** @brief Construct a threadpool of a given size.
   *
   *  @param max_threads Total threads available. max_threads-1 worker
   *                     threads will be launched.
   */
  thread_pool(size_type max_threads);

  /** @brief Destroy the threadpool */
  ~thread_pool() {
    all_work_done_ = true;
    global_work_queue_.wake_all(true);
  }

  /** @brief Launch the threads */
  void launch_threads(size_type num_threads);
  /** @brief Launch the threads and pin them to the Hyperthreaded cores */
  void launch_pinned_threads(size_type num_threads, int cpu_offset);
  /** Wake and terminate all threads in the pool */
  void reap_threads();
  /** Reap all threads in the pool and relaunch pinned threads */
  void relaunch_pinned_threads(size_type num_threads);

  /** @brief Submit a job to the pool's queue */
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

  /** @brief Submit a job to the pool's queue and place the future
      into a work group */
  template <typename FunctionT>
  void submit_job_to_work_group(FunctionT func)
  {
    using return_type = typename std::result_of<FunctionT()>::type;

    std::packaged_task<return_type()> task(std::move(func));
    m_work_group.emplace_back(task.get_future());
    global_work_queue_.push(std::move(task));

    return;
  }

  /** @brief Wait for all of the jobs in a work group to finish */
  bool finish_work_group() {
    std::string error_message;
    for (auto& f : m_work_group) {
      bool valid = f.get();
      if (!valid) {
        error_message = "invalid future in work group";
      }
    }
    m_work_group.clear();
    if (!error_message.empty()) { LBANN_ERROR(error_message); }
    return true;
  }

  /** @brief Query the number of worker threads actually present */
  size_type get_num_threads() const noexcept { return threads_.size(); }

  /** @brief Convert the C++ thread id into a local thread pool id */
  int get_local_thread_id();

  /** @brief Convert the C++ thread id into a local thread pool id */
  int get_threads_offset() { return m_threads_offset; }

private:
  /** @brief The task executed by each thread */
  void do_thread_work_();
  void do_thread_work_pinned_thread_(int tid, cpu_set_t cpu_set);

private:

  /** @brief Container holding the threads */
  thread_container_type threads_;

  /** @brief The thread-safe work queue */
  thread_safe_queue<type_erased_function> global_work_queue_;

  /** @brief RAII "deleter" for the threads */
  thread_joiner thread_joiner_;

  /** @brief Flag to track if more work is to be done */
  std::atomic<bool> all_work_done_;

  std::mutex m_thread_map_mutex;
  std::unordered_map<std::thread::id, int> m_thread_id_to_local_id_map;

  /** @brief Work Group */
  std::vector<std::future<bool>> m_work_group;

  int m_threads_offset;

};// class thread_pool

}// namespace lbann
#endif /* __LBANN_THREAD_POOL_HPP__ */
