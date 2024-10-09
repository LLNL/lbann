////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/logging.hpp"

#include <exception>
#include <iostream>
#include <sstream>

// Macro to throw an LBANN exception
#define LBANN_ERROR(...)                                                       \
  do {                                                                         \
    const int rank_LBANN_ERROR = lbann::get_rank_in_world();                   \
    throw ::lbann::exception(::lbann::build_string(                            \
      "LBANN error",                                                           \
      (rank_LBANN_ERROR >= 0 ? " on rank " + std::to_string(rank_LBANN_ERROR)  \
                             : std::string()),                                 \
      " (",                                                                    \
      __FILE__,                                                                \
      ":",                                                                     \
      __LINE__,                                                                \
      "): ",                                                                   \
      __VA_ARGS__));                                                           \
  } while (0)

// Macro to print a warning to standard error stream.
#define LBANN_WARNING(...) LBANN_WARN(lbann::logging::LBANN_Logger_ID::LOG_RT, __VA_ARGS__)

#define LBANN_WARNING_WORLD_ROOT(...)                                          \
  do {                                                                         \
    if (lbann::get_rank_in_world() == 0) {                                     \
      LBANN_WARNING(__VA_ARGS__);                                              \
    }                                                                          \
  } while (0)

// Macro to print a message to standard cout stream.
#define LBANN_MSG(...)                                                         \
  do {                                                                         \
    const int rank_LBANN_MSG = lbann::get_rank_in_world();                     \
    if (rank_LBANN_MSG == 0) {                                                 \
      std::cout << lbann::build_string(                                        \
                     "LBANN message",                                          \
                     (rank_LBANN_MSG >= 0                                      \
                        ? " on rank " + std::to_string(rank_LBANN_MSG)         \
                        : std::string()),                                      \
                     " (",                                                     \
                     __FILE__,                                                 \
                     ":",                                                      \
                     __LINE__,                                                 \
                     "): ",                                                    \
                     __VA_ARGS__)                                              \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

#define LBANN_ASSERT(cond)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      LBANN_ERROR("The assertion " #cond " failed.");                          \
    }                                                                          \
  } while (0)

#ifdef LBANN_DEBUG
#define LBANN_ASSERT_DEBUG(cond) LBANN_ASSERT(cond)
#else
#define LBANN_ASSERT_DEBUG(cond)
#endif

#define LBANN_ASSERT_WARNING(cond)                                             \
  if (!(cond))                                                                 \
  LBANN_WARNING("The assertion " #cond " failed.")

namespace lbann {

/** @class exception
 *  @brief The base exception for LBANN errors.
 *
 *  A stack trace is recorded when the exception is constructed.
 */
class exception : public std::exception
{
public:
  /** @brief Default constructor.
   *
   *  Uses a generic message that reports the rank and stack trace.
   */
  exception();

  /** @brief Constructor with message.
   *
   *  The message is interpolated into a longer report that includes
   *  the stack trace from where the constructor is called.
   *  Unfortunately, the constructor frame is usually included in
   *  that stack trace.
   */
  exception(std::string message);

  char const* what() const noexcept override;

  /** @brief Print the what() string to the stream. */
  void print_report(std::ostream& os = std::cerr) const;

private:
  /** Human-readable exception message. */
  std::string m_message;
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
std::string build_string(Args&&... args)
{
  std::ostringstream oss;
  int dummy[] = {(oss << args, 0)...};
  (void)dummy; // silence compiler warnings
  return oss.str();
}

} // namespace lbann

#endif // LBANN_UTILS_EXCEPTION_HPP_INCLUDED
