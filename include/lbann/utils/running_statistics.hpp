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
#ifndef LBANN_UTILS_RUNNING_STATISTICS_HPP_INCLUDED
#define LBANN_UTILS_RUNNING_STATISTICS_HPP_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace lbann {

/** @class RunningStats
 *  @brief Accumulate mean, stddev, min, and max over a streaming data
 *         set.
 */
class RunningStats
{
public:
  /** @brief The default minimum value, chosen to ensure that any
   *         observed sample is less-than this.
   */
  constexpr static auto default_min = std::numeric_limits<double>::max();
  /** @brief The default maximum value, chosen to ensure that any
   *         observed sample is greater-than this.
   */
  constexpr static auto default_max = std::numeric_limits<double>::lowest();

public:
  /** @name Modifiers */
  ///@{

  /** @brief All values return to their defaults.
   *
   *  All queries should be considered meaningless unless samples() is
   *  greater than 0.
   */
  void reset() noexcept;

  /** @brief Add a new value to the data set.
   *
   *  The statistics are updated immediately.
   */
  void insert(double val) noexcept;

  ///@}
  /** @name Queries */
  ///@{
  /** @brief Number of observed samples. */
  size_t samples() const noexcept;
  /** @brief Get the minimum observed value.
   *  @note Only valid if samples() > 0.
   */
  double min() const noexcept;
  /** @brief Get the minimum observed value.
   *  @note Only valid if samples() > 0.
   */
  double max() const noexcept;
  /** @brief Running mean of observed samples. */
  double mean() const noexcept;
  /** @brief Running sum of observed samples. */
  double total() const noexcept;
  /** @brief Running (unbiased) sample variance of the observed
   *         samples.
   *  @note This is zero at least until 2 or more samples have been
   *        observed.
   */
  double variance() const noexcept;
  /** @brief Running (unbiased) sample standard deviation of the
   *         observed samples.
   *  @note This is zero at least until 2 or more samples have been
   *        observed.
   */
  double stddev() const noexcept;
  ///@}

private:
  size_t m_count = 0UL;
  double m_min = default_min;
  double m_max = default_max;
  double m_mean = 0.;
  double m_diff_sq = 0.;
}; // class RunningStats

inline void RunningStats::reset() noexcept { *this = RunningStats{}; }

inline void RunningStats::insert(double val) noexcept
{
  // Welford
  ++m_count;
  m_min = std::min(m_min, val);
  m_max = std::max(m_max, val);
  double diff1 = val - m_mean;
  m_mean += diff1 / m_count;
  m_diff_sq += diff1 * (val - m_mean);
}

inline size_t RunningStats::samples() const noexcept { return m_count; }

inline double RunningStats::min() const noexcept { return m_min; }

inline double RunningStats::max() const noexcept { return m_max; }

inline double RunningStats::mean() const noexcept { return m_mean; }

inline double RunningStats::total() const noexcept
{
  return m_mean * static_cast<double>(m_count);
}

inline double RunningStats::variance() const noexcept
{
  return m_count > 1 ? m_diff_sq / static_cast<double>(m_count - 1) : 0.;
}

inline double RunningStats::stddev() const noexcept
{
  return std::sqrt(this->variance());
}

} // namespace lbann
#endif // LBANN_UTILS_RUNNING_STATISTICS_HPP_INCLUDED
