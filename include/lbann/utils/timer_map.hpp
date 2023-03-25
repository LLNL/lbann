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
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_UTILS_TIMER_MAP_HPP_INCLUDED
#define LBANN_UTILS_TIMER_MAP_HPP_INCLUDED

#include "lbann/utils/accumulating_timer.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <string>

namespace lbann {

/** @class TimerMap
 *  @brief A nesting inclusive-timer.
 *
 *  This is useful for timing subsections of an algorithm, for
 *  example. The timing information can be summarized in a visually
 *  structured format. The inclusive time is explicit, and the
 *  exclusive time is easily inferred by subtracting sub-timers'
 *  inclusive time from this inclusive time.
 *
 *  More documentation goes here.
 */
class TimerMap
{
public:
  TimerMap(std::string const& key);

  std::string const& key() const noexcept;
  AccumulatingTimer& timer() noexcept;
  AccumulatingTimer const& timer() const noexcept;

  TimerMap& scope(std::string const& key);
  TimerMap const& scope(std::string const& key) const;

  void print(std::ostream& os) const;

private:
  void print_impl(std::ostream& os, unsigned width, unsigned indent) const;

private:
  std::string m_key;
  AccumulatingTimer m_timer;
  std::list<TimerMap> m_subscopes;

}; // class TimerMap

class ScopeTimer
{
  TimerMap* m_timer;

public:
  ScopeTimer(TimerMap& timer, std::string const& scope_name);
  ScopeTimer(ScopeTimer& timer, std::string const& scope_name);
  ~ScopeTimer() noexcept;
}; // class ScopeTimer

template <typename TimerT>
auto time_scope(TimerT& timer, std::string const& scope_name)
{
  return ScopeTimer{timer, scope_name};
}

// Implementation

inline TimerMap::TimerMap(std::string const& key) : m_key{key} {}

inline std::string const& TimerMap::key() const noexcept { return m_key; }
inline AccumulatingTimer& TimerMap::timer() noexcept { return m_timer; }
inline AccumulatingTimer const& TimerMap::timer() const noexcept
{
  return m_timer;
}

inline auto TimerMap::scope(std::string const& key) -> TimerMap&
{
  auto iter = std::find_if(begin(m_subscopes),
                           end(m_subscopes),
                           [&key](auto& t) { return t.key() == key; });
  if (iter == end(m_subscopes))
    return m_subscopes.emplace_back(key);
  else
    return *iter;
}

inline ScopeTimer::ScopeTimer(TimerMap& timer, std::string const& scope_name)
  : m_timer{&(timer.scope(scope_name))}
{
  m_timer->timer().start();
}

inline ScopeTimer::ScopeTimer(ScopeTimer& timer, std::string const& scope_name)
  : ScopeTimer{*(timer.m_timer), scope_name}
{}

inline ScopeTimer::~ScopeTimer() noexcept
{
  m_timer->timer().stop();
  m_timer = nullptr;
}

} // namespace lbann
#endif // LBANN_UTILS_TIMER_MAP_HPP_INCLUDED
