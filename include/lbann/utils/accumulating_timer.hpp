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

#ifndef LBANN_UTILS_ACCUMULATING_TIMER_INCLUDED
#define LBANN_UTILS_ACCUMULATING_TIMER_INCLUDED

#include "lbann/utils/running_statistics.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

/** @class AccumulatingTimer
 *  @brief Timer that accumulates mean and variance of timer durations
 *         as it goes.
 */
class AccumulatingTimer
{
public:
  /** @name Timing interface */
  ///@{

  /** @brief Start counting time for this duration sample.
   *
   *  If the clock is already running, this will restart the counter.
   */
  void start() noexcept;

  /** @brief Get the elapsed time for this duration in seconds.
   *
   *  Resets the counter, so a subsequent call to stop() or check()
   *  without another start() will return 0.0.
   *
   *  The time will be committed to the running statistics
   *  accumulation only if the timer is in the "running" state when
   *  this is called.
   */
  double stop() noexcept;

  /** @brief Get the current elapsed time in this duration (in
   *         seconds) without stopping the timer.
   */
  double check() const noexcept;

  /** @brief Reset this duration without caching it in the running
   *         statistics.
   */
  void reset() noexcept;

  /** @brief Determine whether there is an active duration sample running. */
  bool running() const noexcept;

  ///@}
  /** @name Statistics */
  ///@{

  /** @brief The number of durations observed in this timer. */
  size_t samples() const noexcept;

  /** @brief The mean observed duration. */
  double mean() const noexcept;

  /** @brief The standard deviation of observed durations. */
  double stddev() const noexcept;

  /** @brief The smallest observed duration. */
  double min() const noexcept;

  /** @brief The largest observed duration. */
  double max() const noexcept;

  /** @brief The total time observed by this timer.
   *
   *  Only time that has been committed is reported. That is, it does
   *  NOT include any time from the current sample if the timer is
   *  currently running.
   */
  double total_time() const noexcept;

  /** @brief Clear the running history of durations. */
  void reset_statistics() noexcept;

  ///@}
private:
  Timer m_timer;
  RunningStats m_stats;
}; // class AccumulatingTimer

inline void AccumulatingTimer::start() noexcept { m_timer.start(); }

inline double AccumulatingTimer::stop() noexcept
{
  if (running()) {
    auto elapsed_time = m_timer.stop();
    m_stats.insert(elapsed_time);
    return elapsed_time;
  }
  return 0.;
}

inline double AccumulatingTimer::check() const noexcept
{
  return m_timer.check();
}

inline void AccumulatingTimer::reset() noexcept { m_timer.reset(); }

inline bool AccumulatingTimer::running() const noexcept
{
  return m_timer.running();
}

inline size_t AccumulatingTimer::samples() const noexcept
{
  return m_stats.samples();
}

inline double AccumulatingTimer::mean() const noexcept
{
  return m_stats.mean();
}

inline double AccumulatingTimer::stddev() const noexcept
{
  return m_stats.stddev();
}

inline double AccumulatingTimer::min() const noexcept
{
  return m_stats.samples() ? m_stats.min() : 0.;
}

inline double AccumulatingTimer::max() const noexcept
{
  return m_stats.samples() ? m_stats.max() : 0.;
}

inline double AccumulatingTimer::total_time() const noexcept
{
  return m_stats.total();
}

inline void AccumulatingTimer::reset_statistics() noexcept { m_stats.reset(); }

} // namespace lbann
#endif // LBANN_UTILS_ACCUMULATING_TIMER_INCLUDED
