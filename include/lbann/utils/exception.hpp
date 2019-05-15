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
#include <iostream>
#include <sstream>
#include <exception>

// Macro to throw an LBANN exception
#define LBANN_ERROR(message)                                    \
  do {                                                          \
    std::stringstream ss_LBANN_ERROR;                           \
    ss_LBANN_ERROR << "LBANN error ";                           \
    const int rank_LBANN_ERROR = lbann::get_rank_in_world();    \
    if (rank_LBANN_ERROR >= 0) {                                \
      ss_LBANN_ERROR << "on rank " << rank_LBANN_ERROR << " ";  \
    }                                                           \
    ss_LBANN_ERROR << "(" << __FILE__ << ":" << __LINE__ << ")" \
                     << ": " << (message);                      \
    throw lbann::exception(ss_LBANN_ERROR.str());               \
  } while (0)

// Macro to print a warning to standard error stream.
#define LBANN_WARNING(message)                                          \
  do {                                                                  \
    std::stringstream ss_LBANN_WARNING;                                 \
    ss_LBANN_WARNING << "LBANN warning ";                               \
    const int rank_LBANN_WARNING = lbann::get_rank_in_world();          \
    if (rank_LBANN_WARNING >= 0) {                                      \
      ss_LBANN_WARNING << "on rank " << rank_LBANN_WARNING << " ";      \
    }                                                                   \
    ss_LBANN_WARNING << "(" << __FILE__ << ":" << __LINE__ << ")"       \
                     << ": " << (message) << std::endl;                 \
    std::cerr << ss_LBANN_WARNING.str();                                \
  } while (0)

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

} // namespace lbann

#endif // LBANN_UTILS_EXCEPTION_HPP_INCLUDED
