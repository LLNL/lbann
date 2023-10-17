////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
//
// lbann_timer .hpp - Wrapper around time functionality
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_UTILS_TIMER_HPP_INCLUDED
#define LBANN_UTILS_TIMER_HPP_INCLUDED

#include <chrono>

namespace lbann {

// Datatype for model evaluation
// Examples: timing, metrics, objective functions
#ifdef LBANN_HAS_DOUBLE
using EvalType = double;
#else
using EvalType = float;
#endif // LBANN_HAS_DOUBLE

/** @brief Return time in fractional seconds since an epoch. */
// inline double get_time()
inline EvalType get_time()
{
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch())
    .count();
}

/** @class Timer
 *  @brief An exceedingly simple duration calculator.
 *
 *  This clock does not have a notion of "pause and restart". Calling
 *  check() will not stop the counter; calling stop() will clear the
 *  counter. Calling start() on a running timer will reset the clock
 *  to zero.
 */
class Timer
{
public:
  /** @brief The clock used for counting.
   *
   *  Per guidance from cppreference, this should be the steady clock
   *  for measuring durations. The high-res clock can be unstable.
   *  ([Source](https://en.cppreference.com/w/cpp/chrono/high_resolution_clock))
   */
  using ClockType = std::chrono::steady_clock;
  /** @brief Simplifying typedef. */
  using TimePoint = typename ClockType::time_point;

public:
  Timer() = default;
  ~Timer() noexcept = default;
  Timer(Timer const&) = delete;
  Timer& operator=(Timer const&) = delete;
  Timer(Timer&&) = default;
  Timer& operator=(Timer&&) = default;

  /** @brief Start counting time.
   *
   *  If the clock is already running, this will restart the counter.
   *
   *  @post running() returns @c true; stop() or check() will return a
   *        positive value.
   */
  void start() noexcept { m_start = ClockType::now(); }

  /** @brief Get the total elapsed time in seconds.
   *
   *  Resets the counter, so a subsequent call to stop() or check()
   *  without another start() will return 0.0.
   *
   *  @post running() returns @c false, as if reset() were called.
   */
  //  double stop() noexcept
  EvalType stop() noexcept
  {
    auto elapsed_time = this->check();
    this->reset();
    return elapsed_time;
  }

  /** @brief Get the current elapsed time (seconds) without stopping. */
  EvalType check() const noexcept
  //  double check() const noexcept
  {
    using seconds_type = std::chrono::duration<double>;
    return running() ? seconds_type(ClockType::now() - m_start).count() : 0.;
  }

  /** @brief Clear any internal state in the timer.
   *  @post running() returns @c false; check() and stop() return 0.
   */
  void reset() noexcept { m_start = TimePoint{}; }

  /** @brief Check if the timer is running. */
  bool running() const noexcept { return m_start != TimePoint{}; }

private:
  TimePoint m_start;

}; // class Timer

} // namespace lbann

#endif // LBANN_UTILS_TIMER_HPP_INCLUDED
