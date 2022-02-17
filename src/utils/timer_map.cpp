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

#include "lbann/utils/timer_map.hpp"

#include "lbann/utils/accumulating_timer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/output_helpers.hpp"

#include <ios>
#include <iostream>
#include <sstream>

namespace lbann {

auto TimerMap::scope(std::string const& key) const -> TimerMap const&
{
  auto iter = std::find_if(begin(m_subscopes),
                           end(m_subscopes),
                           [&key](auto& t) { return t.key() == key; });
  if (iter == end(m_subscopes))
    LBANN_ERROR("No scope \"", key, "\" exists in this TimerMap.");
  return *iter;
}

static void print_timer(std::ostream& os,
                        std::string const& k,
                        AccumulatingTimer const& t,
                        unsigned const width,
                        unsigned const indent)
{
  // clang-format off
  auto const key_width = width - 65U - indent;
  os << std::string(indent, ' ')
     << std::setw(key_width) << std::left
     << truncate_to_width(k, key_width) << " | "
     << std::setw(8) << std::right << t.samples() << " | "
     << std::scientific << std::setprecision(4)
     << std::setw(10) << std::right << t.total_time() << " | "
     << std::setw(10) << std::right << t.mean() << " | "
     << std::setw(10) << std::right << t.min() << " | "
     << std::setw(10) << std::right << t.max() << " |\n";
  // clang-format on
}

static unsigned output_width(std::ostream& os)
{
  // If 0, set it to 100. Min of 75.
  unsigned const term_width = get_window_size(os).second;
  return (term_width ? std::min(std::max(75U, term_width), 120U) : 100U);
}

static void
print_header(std::ostream& os, std::string const& name, unsigned const width)
{
  // clang-format off
  auto const key_width = width - 65U;
  os << std::string(width, '=') << "\n"
     << cyan << "Timer: " << nocolor << name << "\n"
     << red << std::setw(key_width) << std::left << "Label" << nocolor << " | "
     << red << std::setw(8) << std::left << "Samples" << nocolor << " | "
     << red << std::setw(10) << std::left << "Total" << nocolor << " | "
     << red << std::setw(10) << std::left << "Mean" << nocolor << " | "
     << red << std::setw(10) << std::left << "Min" << nocolor << " | "
     << red << std::setw(10) << std::left << "Max" << nocolor << " |\n"
     << std::string(key_width, '-')
     << "-|----------|------------|------------|------------|------------|\n";
  // clang-format off
}

void TimerMap::print(std::ostream& os_in) const
{
  std::ostringstream oss;
  unsigned const width = output_width(os_in);
  bool const with_color = is_good_terminal(os_in);
  std::ostream& os = (with_color ? os_in : oss);

  print_header(os, this->key(), width);
  this->print_impl(os, width, 0);
  if (!with_color)
    os_in << strip_ansi_csis(oss.str());

  // print_footer()
  os_in << std::string(width, '=') << "\n";
}

void TimerMap::print_impl(std::ostream& os,
                          unsigned const width,
                          unsigned const indent) const
{
  if (this->timer().samples())
    print_timer(os, this->key(), this->timer(), width, indent);
  for (auto const& t : m_subscopes)
    t.print_impl(os, width, indent + 1);
}

} // namespace lbann
