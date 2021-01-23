////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "ReplaceEscapes.hpp"

#include <lbann/utils/environment_variable.hpp>
#include <lbann/utils/system_info.hpp>

#include <regex>
#include <stdexcept>
#include <string>

namespace unit_test
{
namespace utilities
{

namespace
{

std::string GetBasicReplacement(
  std::string const& str, lbann::utils::SystemInfo const& system_info)
{
  if (str.size() != 2 || str[0] != '%')
    throw std::logic_error("string is not a valid pattern.");

  switch (str[1])
  {
  case 'h':
    return system_info.host_name();
  case 'p':
    return system_info.pid();
  case 'r':
    return std::to_string(system_info.mpi_rank());
  case 's':
    return std::to_string(system_info.mpi_size());
  default:
    throw BadSubstitutionPattern(str);
  }
  return ""; // in case a compiler complains about no return.
}

}// namespace <anon>

BadSubstitutionPattern::BadSubstitutionPattern(std::string const& str)
  : std::runtime_error("Bad escape sequence: " + str)
{}

std::string replace_escapes(
  std::string const& str, lbann::utils::SystemInfo const& system_info)
{
  std::regex re("%env\\{([a-zA-Z0-9_]+)}|%[a-zA-Z]", std::regex::extended);
  std::smatch match;
  std::string outstr;
  outstr.reserve(str.size());
  size_t start=0;

  do
  {
    // Get the string up to the first %%
    auto const end = str.find("%%", start);
    auto tmp = str.substr(start, end-start);

    // Do all replacements
    while (regex_search(tmp, match, re))
    {
      if (match.size() != 2UL)
        throw std::logic_error("Unexpected match size");

      if (match[1].length() == 0)
        tmp.replace(match.position(), match.length(),
                    GetBasicReplacement(match[0], system_info));
      else
        tmp.replace(match.position(), match.length(),
                    system_info.env_variable_value(match[1]));
    }
    outstr += tmp + "%";

    // Update the starting position in the original string.
    start = (end == std::string::npos) ? std::string::npos : end+2;

  }
  while (start != std::string::npos);

  // Added an extra "%"; remove it.
  outstr.pop_back();

  return outstr;
}

}// namespace utilities
}// namespace unit_test
