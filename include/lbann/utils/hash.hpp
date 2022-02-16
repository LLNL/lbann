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

#ifndef LBANN_UTILS_HASH_HPP_INCLUDED
#define LBANN_UTILS_HASH_HPP_INCLUDED

#include <functional>
#include <type_traits>
#include <utility>

namespace lbann {

/** @brief Combine two hash values
 *
 *  A hash function is applied to an object and the resulting hash
 *  value is mixed with another hash value. See
 *  https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine.
 *
 *  @param seed     Hash value.
 *  @param val      Input to hash function.
 *  @tparam Hash    Hash function for type @c T.
 */
template <class T, class Hash=std::hash<T>>
std::size_t hash_combine(std::size_t seed, const T& val) {
  return seed ^ (Hash()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/** @brief Hash function for enumeration type
 *
 *  Equivalent to @c std::hash if the input is not an enumeration
 *  type.
 */
template <class T>
struct enum_hash {
  using underlying_t
  = typename std::conditional<std::is_enum<T>::value,
                              typename std::underlying_type<T>::type,
                              T>::type;
  std::size_t operator()(T val) const {
    return std::hash<underlying_t>()(static_cast<underlying_t>(val));
  }
};

/** @brief Hash function for @c std::pair */
template <class T1,
          class T2,
          class Hash1=std::hash<T1>,
          class Hash2=std::hash<T2>>
struct pair_hash {
  std::size_t operator()(const std::pair<T1,T2>& val) const {
    auto seed = Hash1()(val.first);
    return hash_combine<T2,Hash2>(seed, val.second);
  }
};

} // namespace lbann

#endif // LBANN_UTILS_HASH_HPP_INCLUDED
