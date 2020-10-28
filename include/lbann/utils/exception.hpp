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

#ifndef LBANN_UTILS_EXCEPTION_HPP_INCLUDED
#define LBANN_UTILS_EXCEPTION_HPP_INCLUDED

#include "lbann/comm.hpp"

#include <exception>
#include <iostream>
#include <sstream>

// Macro to throw an LBANN exception
#define LBANN_ERROR(...)                                        \
  do {                                                          \
    const int rank_LBANN_ERROR = lbann::get_rank_in_world();    \
    throw lbann::exception(                                     \
      lbann::build_string(                                      \
        "LBANN error",                                          \
        (rank_LBANN_ERROR >= 0                                  \
         ? " on rank " + std::to_string(rank_LBANN_ERROR)       \
         : std::string()),                                      \
        " (", __FILE__, ":", __LINE__, "): ", __VA_ARGS__));    \
  } while (0)

// Macro to print a warning to standard error stream.
#define LBANN_WARNING(...)                                      \
  do {                                                          \
    const int rank_LBANN_WARNING = lbann::get_rank_in_world();  \
    std::cerr << lbann::build_string(                           \
      "LBANN warning",                                          \
      (rank_LBANN_WARNING >= 0                                  \
       ? " on rank " + std::to_string(rank_LBANN_WARNING)       \
       : std::string()),                                        \
      " (", __FILE__, ":", __LINE__, "): ", __VA_ARGS__)        \
              << std::endl;                                     \
  } while (0)

// Macro to print a message to standard cout stream.
#define LBANN_MSG(...)                                          \
  do {                                                          \
    const int rank_LBANN_MSG = lbann::get_rank_in_world();      \
    if(rank_LBANN_MSG == 0) {                                   \
      std::cout << lbann::build_string(                         \
      "LBANN message",                                          \
      (rank_LBANN_MSG >= 0                                      \
       ? " on rank " + std::to_string(rank_LBANN_MSG)           \
       : std::string()),                                        \
      " (", __FILE__, ":", __LINE__, "): ", __VA_ARGS__)        \
              << std::endl;                                     \
    }                                                           \
  } while (0)

#define LBANN_ASSERT(cond)                              \
  if (!(cond))                                          \
    LBANN_ERROR("The assertion " #cond " failed.")

#define LBANN_ASSERT_WARNING(cond)                      \
  if (!(cond))                                          \
    LBANN_WARNING("The assertion " #cond " failed.")

namespace lbann {

/** Exception.
 *  A stack trace is recorded when the exception is constructed.
 */
class exception : public std::exception {
public:

  /** Constructor.
   *  By default, a human-readable report is immediately printed to
   *  the standard error stream.
   */
  exception(std::string message = "", bool print = true);
  const char* what() const noexcept override;

  /** Print human-readable report to stream.
   *  Reports the exception message and the stack trace.
   */
  void print_report(std::ostream& os = std::cerr) const;

private:
  /** Human-readable exception message. */
  std::string m_message;
  /** Human-readable stack trace.
   *  The stack trace is recorded when the exception is constructed.
   */
  std::string m_stack_trace;

};
using lbann_exception = exception;

/** @brief Build a string from the arguments.
 *
 *  The arguments must be stream-outputable (have operator<<(ostream&,
 *  T) defined). It will be a static error if this fails.
 *
 *  @tparam Args (Inferred) The types of the arguments.
 *
 *  @param[in] args The things to be stringified.
 */
template <typename... Args>
std::string build_string(Args&&... args) {
  std::ostringstream oss;
  int dummy[] = { (oss << args, 0)... };
  (void) dummy; // silence compiler warnings
  return oss.str();
}

} // namespace lbann

#endif // LBANN_UTILS_EXCEPTION_HPP_INCLUDED
