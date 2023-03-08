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
#ifndef LBANN_UTILS_THREADS_TYPE_ERASED_FUNCTION_HPP_INCLUDED
#define LBANN_UTILS_THREADS_TYPE_ERASED_FUNCTION_HPP_INCLUDED

#include <lbann/utils/memory.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace lbann {

/** @class type_erased_function
 *  @brief A move-only callable type for wrapping functions
 */
class type_erased_function
{
public:
  /** @brief Erase the type of input function F */
  template <typename FunctionT>
  type_erased_function(FunctionT&& F)
    : held_function_(make_unique<Function<FunctionT>>(std::move(F)))
  {}

  /** @brief Move constructor */
  type_erased_function(type_erased_function&& other) = default;

  /** @brief Move assignment */
  type_erased_function& operator=(type_erased_function&& other) = default;

  /** @brief Make the function callable */
  void operator()() { held_function_->call_held(); }

  /** @name Deleted functions */
  ///@{

  /** @brief Deleted constructor */
  type_erased_function() = delete;

  /** @brief Deleted copy constructor */
  type_erased_function(const type_erased_function& other) = delete;

  /** @brief Deleted copy assignment */
  type_erased_function& operator=(const type_erased_function& other) = delete;

  ///@}

private:
  /** @name Type erasure template types */
  ///@{

  /** @class FunctionHolder
   *  @brief Simple function object holder
   */
  struct FunctionHolder
  {
    /** @brief Destructor */
    virtual ~FunctionHolder() = default;

    /** @brief Call the held function */
    virtual void call_held() = 0;
  };

  /** @class Function
   *  @brief A wrapper for a specific type of function
   *
   *  @tparam FunctionT Must be MoveConstructible and Callable
   */
  template <typename FunctionT>
  struct Function : FunctionHolder
  {
    static_assert(std::is_move_constructible<FunctionT>::value,
                  "Given type is not move constructible!");

    /** @brief Construct by moving from the input function type */
    Function(FunctionT&& f) : F__(std::move(f)) {}

    /** @brief Destructor */
    ~Function() = default;

    /** @brief Call the held function */
    void call_held() override { F__(); }

    /** @brief The held function */
    FunctionT F__;
  };
  ///@}

  /** @brief A type-erased function */
  std::unique_ptr<FunctionHolder> held_function_;
}; // class type_erased_function

} // namespace lbann
#endif /* LBANN_UTILS_THREADS_TYPE_ERASED_FUNCTION_HPP_INCLUDED */
