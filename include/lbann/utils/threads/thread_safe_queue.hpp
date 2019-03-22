#ifndef __LBANN_THREAD_SAFE_QUEUE_HPP__
#define __LBANN_THREAD_SAFE_QUEUE_HPP__

#include <condition_variable>
#include <mutex>

#include <lbann/utils/memory.hpp>

namespace lbann {

/** @class thread_safe_queue
 *  @brief A queue that is safe for multiple threads to push to or
 *  pull from "simultaneously".
 *
 *  This version uses locks.
 *
 *  This is essentially a fancy linked-list implementation that
 *  enables finer-grained locks than simply wrapping an
 *  std::queue. The trade-off is two locks, one for the front and one
 *  for the back of the list.
 *
 *  @tparam T A move- or copy-constructible type
 */
template <typename T>
class thread_safe_queue {
private:

  /** @class _Node
   *  @brief A data value in the thread-safe FIFO queue
   */
  struct _Node
  {
    std::unique_ptr<T> data_;
    std::unique_ptr<_Node> next_;
  };

public:

  /** @brief Default constructor; creates an empty queue */
  thread_safe_queue()
    : head_(make_unique<_Node>()), tail_(head_.get()), m_stop_threads(false)
  {}

  /** @brief Adds a value to back of the queue */
  void push(T value)
  {
    // Make the new data outside of the lock to minimize lock time
    auto new_value = make_unique<T>(std::move(value));
    auto new_node = make_unique<_Node>();

    // Adding to the queue only modifies the tail
    {
      std::lock_guard<std::mutex> lk(tail_mtx_);
      tail_->data_ = std::move(new_value);
      tail_->next_ = std::move(new_node);
      tail_ = tail_->next_.get();
    }
    // Update the condition variable (for wait_and_pop)
    data_available_.notify_one();
  }

  void wake_all(bool stop = false) {
    {
      std::lock_guard<std::mutex> lk(head_mtx_);
      m_stop_threads = stop;
    }
    // Update the condition variable (for wait_and_pop)
    data_available_.notify_all();
  }

  /// Allow the thread pool to set / reset the flags
  void set_stop_threads(bool flag) { m_stop_threads = flag; }

  /** @brief Try to remove the first value from the queue
   *
   *  @return nullptr if empty(); otherwise return a value
   */
  std::unique_ptr<T> try_pop()
  {
    std::unique_lock<std::mutex> lk(head_mtx_);
    if (head_.get() == do_get_tail_()) return nullptr;

    // Remove the head
    auto popped_head = std::move(head_);
    head_ = std::move(popped_head->next_);

    return std::move(popped_head->data_);
  }

  /** @brief Wait for data and then return it */
  std::unique_ptr<T> wait_and_pop()
  {
    std::unique_lock<std::mutex> lk(head_mtx_);
    data_available_.wait(lk,[&]{return ((head_.get() != do_get_tail_())
                                        || ((head_.get() ==
                                             do_get_tail_()) &&
                                            m_stop_threads ));});

    // There is no more work to do, bail
    if(head_.get() == do_get_tail_() && m_stop_threads) {
      return nullptr;
    }

    // Remove the head
    auto popped_head = std::move(head_);
    head_ = std::move(popped_head->next_);

    return std::move(popped_head->data_);
  }

  /** @brief Check if queue is empty */
  bool empty() const
  {
    std::lock_guard<std::mutex> lk(head_mtx_);
    return (head_.get() == do_get_tail_());
  }

private:

  /** @brief Get the tail pointer */
  _Node* do_get_tail_() const
  {
    std::lock_guard<std::mutex> lk(tail_mtx_);
    return tail_;
  }

private:

  /** @brief The mutex protecting the head of the list */
  mutable std::mutex head_mtx_;

  /** @brief The mutex protecting the tail of the list */
  mutable std::mutex tail_mtx_;

  /** @brief The first node in the list */
  std::unique_ptr<_Node> head_;

  /** @brief The last node in the list */
  _Node* tail_;

  /** @brief Condition variable tripped when data added */
  std::condition_variable data_available_;

  bool m_stop_threads;

};// class thread_safe_queue

}// namespace lbann
#endif /* __LBANN_THREAD_SAFE_QUEUE_HPP__ */
