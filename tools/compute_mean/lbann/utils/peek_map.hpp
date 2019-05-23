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
//
// peek_map .hpp .cpp - Utility to peek into a map without the side effect
//                      of adding the default value when a key is not found
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_PEEK_MAP_HPP_INCLUDED
#define LBANN_PEEK_MAP_HPP_INCLUDED

#include <map>

namespace lbann {

/**
 * A utility function to peek into a std::map without the side effect of adding
 * the default value when a key is not found.
 * It has the the template parameter list as std::map does except the allocator.
 * Searches 'idx' in std::map 'map_to_peek', and returns the matching value if
 * found or the default otherwise. 'found' is set to true when the key 'idx' is
 * found, or false when not.
 */
template < class KEY_T,                    // map::key_type
           class VAL_T,                    // map::mapped_type
           class CMP_T = std::less<KEY_T>  // map::key_compare
         >
VAL_T peek_map(const std::map<KEY_T, VAL_T, CMP_T>& map_to_peek, KEY_T idx, bool& found) {
  using map_to_peek_t = std::map<KEY_T, VAL_T, CMP_T>;
  typename map_to_peek_t::const_iterator it = map_to_peek.find(idx);
  if (it == map_to_peek.cend()) {
    found = false;
    return VAL_T();
  }
  found = true;
  return it->second;
}

/**
 * Same as the other peek_map interface but does not require the third argument
 * that indicates the success of searching.
 */
template < class KEY_T,                    // map::key_type
           class VAL_T,                    // map::mapped_type
           class CMP_T = std::less<KEY_T>  // map::key_compare
         >
VAL_T peek_map(const std::map<KEY_T, VAL_T, CMP_T>& map_to_peek, KEY_T idx) {
  using map_to_peek_t = std::map<KEY_T, VAL_T, CMP_T>;
  typename map_to_peek_t::const_iterator it = map_to_peek.find(idx);
  if (it == map_to_peek.cend()) {
    return VAL_T();
  }
  return it->second;
}

/**
 * A utility function to peek into a std::unordered_map without the side effect of adding
 * the default value when a key is not found.
 * It has the the template parameter list as std::unordered_map does except the allocator.
 * Searches 'idx' in std::unordered_map 'map_to_peek', and returns the matching value if
 * found or the default otherwise. 'found' is set to true when the key 'idx' is
 * found, or false when not.
 */
template < class KEY_T,                         // unordered_map::key_type
	   class VAL_T,                         // unordered_map::mapped_type
           class HASH_T = std::hash<KEY_T>,     // unordered_map::hasher
           class KEYeq_T = std::equal_to<KEY_T> // unordered_map::key_equal
	 >
VAL_T peek_map(const std::unordered_map<KEY_T, VAL_T, HASH_T, KEYeq_T>& map_to_peek, KEY_T idx, bool& found) {
  using map_to_peek_t = std::unordered_map<KEY_T, VAL_T, HASH_T, KEYeq_T>;
  typename map_to_peek_t::const_iterator it = map_to_peek.find(idx);
  if (it == map_to_peek.cend()) {
    found = false;
    return VAL_T();
  }
  found = true;
  return it->second;
}

/**
 * Same as the other peek_map interface but does not require the third argument
 * that indicates the success of searching.
 */
template < class KEY_T,                         // unordered_map::key_type
	   class VAL_T,                         // unordered_map::mapped_type
           class HASH_T = std::hash<KEY_T>,     // unordered_map::hasher
           class KEYeq_T = std::equal_to<KEY_T> // unordered_map::key_equal
	 >
VAL_T peek_map(const std::unordered_map<KEY_T, VAL_T, HASH_T, KEYeq_T>& map_to_peek, KEY_T idx) {
  using map_to_peek_t = std::unordered_map<KEY_T, VAL_T, HASH_T, KEYeq_T>;
  typename map_to_peek_t::const_iterator it = map_to_peek.find(idx);
  if (it == map_to_peek.cend()) {
    return VAL_T();
  }
  return it->second;
}

} // end of namespace

#endif // LBANN_PEEK_MAP_HPP_INCLUDED
