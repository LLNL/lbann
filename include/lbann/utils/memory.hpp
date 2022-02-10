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
#ifndef LBANN_MEMORY_HPP_
#define LBANN_MEMORY_HPP_

#include <lbann_config.hpp>
#include <memory>

namespace lbann {

#ifdef LBANN_HAS_STD_MAKE_UNIQUE

using ::std::make_unique;

#else

/** @brief Local definition of make_unique for non-C++14 compilers.
 *  @ingroup stl_wrappers
 */
template <typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

#endif

/** @brief Convert the raw pointer to a unique_ptr. */
template <typename T>
std::unique_ptr<T> to_unique_ptr(T* ptr)
{
  return std::unique_ptr<T>(ptr);
}

}// namespace lbann

#endif /* LBANN_MEMORY_HPP_ */
