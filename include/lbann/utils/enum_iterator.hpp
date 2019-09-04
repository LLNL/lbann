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

#ifndef LBANN_ENUM_ITERATOR_H
#define LBANN_ENUM_ITERATOR_H

#include <type_traits>

namespace lbann {

template < typename C, C beginVal, C endVal>
class EnumIterator {
  typedef typename std::underlying_type<C>::type val_t;
  int val;
public:
  EnumIterator(const C & f) : val(static_cast<val_t>(f)) {}
  EnumIterator() : val(static_cast<val_t>(beginVal)) {}
  EnumIterator operator++() {
    ++val;
    return *this;
  }
  C operator*() { return static_cast<C>(val); }
  EnumIterator begin() { return *this; } //default ctor is good
  EnumIterator end() {
      static const EnumIterator endIter=++EnumIterator(endVal); // cache it
      return endIter;
  }
  bool operator!=(const EnumIterator& i) { return val != i.val; }
};

}
#endif // LBANN_ENUM_ITERATOR_H
