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

#ifndef LBANN_UTILS_FROM_STRING_INCLUDED
#define LBANN_UTILS_FROM_STRING_INCLUDED

#include <algorithm>
#include <string>

namespace lbann {
namespace utils {

/** @brief An exceedingly simple implementation of boost::lexical_cast, e.g.
 *
 *  Generally, these implementations prefer `sto*` function calls to
 *  the stream method because stream operators do not provide
 *  straight-forward error feedback.
 *
 *  @tparam T The type to cast to.
 *
 *  @param str The input string.
 *
 *  @return The value of the input string as a T.
 *
 *  @todo chars, shorts, unsigned. Bool needs some work.
 */
template <typename T>
T from_string(std::string const& str);

inline std::string from_string(std::string&& str)
{
  return std::move(str);
}

template <>
inline std::string from_string<std::string>(std::string const& str)
{
  return str;
}

template <>
inline int from_string<int>(std::string const& str)
{
  return std::stoi(str);
}

template <>
inline long from_string<long>(std::string const& str)
{
  return std::stol(str);
}

template <>
inline long long from_string<long long>(std::string const& str)
{
  return std::stoll(str);
}

template <>
inline unsigned long from_string<unsigned long>(std::string const& str)
{
  return std::stoul(str);
}

template <>
inline unsigned long long from_string<unsigned long long>(std::string const& str)
{
  return std::stoull(str);
}

template <>
inline float from_string<float>(std::string const& str)
{
  return std::stof(str);
}

template <>
inline double from_string<double>(std::string const& str)
{
  return std::stod(str);
}

template <>
inline long double from_string<long double>(std::string const& str)
{
  return std::stold(str);
}

template <>
inline bool from_string<bool>(std::string const& str)
{
  auto upcase = [](std::string s) {
                  std::transform(s.begin(), s.end(), s.begin(),
                                 [](unsigned char c)
                                 { return std::toupper(c); });
                  return s;
                };
  auto upper = upcase(str);
  if (upper == "TRUE")
    return true;
  else if (upper == "FALSE")
    return false;
  else
    return from_string<int>(str);
}

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_FROM_STRING_INCLUDED
