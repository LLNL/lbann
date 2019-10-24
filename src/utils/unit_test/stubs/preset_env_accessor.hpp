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

#ifndef LBANN_UTILS_STUBS_PRESET_ENV_ACCESSOR_HPP_INCLUDED
#define LBANN_UTILS_STUBS_PRESET_ENV_ACCESSOR_HPP_INCLUDED

#include <string>
#include <unordered_map>

namespace lbann {
namespace utils {
namespace stubs {

class PresetEnvAccessor
{
public:
  std::string get(std::string const&) const;
private:
  static void populate_vars();
private:
  static std::unordered_map<std::string, std::string> vars_;
};

inline std::string PresetEnvAccessor::get(std::string const& var_name) const
{
  if (vars_.size() == 0UL) populate_vars();

  auto it = vars_.find(var_name);
  if (it == vars_.end())
    return "";

  return it->second;
}

inline void PresetEnvAccessor::populate_vars()
{
  vars_ = {
    {"APPLE", "3.14"}, // float
    {"ICE_CREAM_SCOOPS", "3"}, // int
    {"PIZZA", "pepperoni"} // string
  };
}

}// namespace stubs
}// namespace utils
}// namespace lbann

#endif /* LBANN_UTILS_STUBS_PRESET_ENV_ACCESSOR_HPP_INCLUDED */
