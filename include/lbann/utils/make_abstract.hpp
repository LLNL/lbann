////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UTILS_MAKE_ABSTRACT_HPP_INCLUDED
#define LBANN_UTILS_MAKE_ABSTRACT_HPP_INCLUDED

#include <google/protobuf/message.h>
#include <type_traits>

namespace lbann {
/** @brief Template that always has a 'value' field that evaluates to
 *         'false'.
 *
 *  This is mostly for static_asserts so the literal "false" isn't in
 *  the "expression" argument. Some compilers also give
 */
template <typename T> struct False : std::false_type
{
};

template <class BaseClass>
std::unique_ptr<BaseClass> make_abstract(google::protobuf::Message const& msg)
{
  static_assert(False<BaseClass>::value,
                "There is no specialization of make_abstract() for this type.");
  return nullptr; // silence compiler warnings
}

template <class ConcreteClass>
std::unique_ptr<ConcreteClass> make(google::protobuf::Message const& msg)
{
  static_assert(False<ConcreteClass>::value,
                "There is no specialization of make() for this type.");
  return nullptr; // silence compiler warnings
}

} // namespace lbann
#endif // LBANN_UTILS_MAKE_ABSTRACT_HPP_INCLUDED
