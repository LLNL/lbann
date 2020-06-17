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

#ifndef LBANN_UNIT_TEST_UTILITIES_REPLACE_ESCAPES_HPP_INCLUDED
#define LBANN_UNIT_TEST_UTILITIES_REPLACE_ESCAPES_HPP_INCLUDED

#include <lbann/utils/system_info.hpp>

#include <stdexcept>
#include <string>

namespace unit_test
{
namespace utilities
{

//
// NOTE TO C++ READERS: The following documentation will appear WRONG
// to you, but it is not! DO NOT CHANGE THE PATTERN/REPLACEMENT TABLE!
// There are many extra percent signs, but these are necessary for the
// markdown to render the HTML correctly! For your benefit, the valid
// sequences are:
//
// %% -- A literal percent sign
// %h -- The hostname of the current process
// %p -- The PID of the current process
// %r -- the MPI rank of the current process, if available, or 0
// %s -- the MPI size of the current job, if available, or 1
// %env{NAME} -- The value of ${NAME} in the current environment
//

/** @brief Substitute basic escape sequences in a string.
 *
 *  The following patterns are supported:
 *
 *  Pattern         | Replacement
 *  --------------- | -----------
 *  %%              | A literal percent sign ("%")
 *  %%h             | The hostname of the current process
 *  %%p             | The PID of the current process
 *  %%r             | The MPI rank of the current process, if available, or 0
 *  %%s             | The MPI size of the current job, if available, or 1
 *  %%env{\<NAME\>} | The value of ${NAME} in the current environment
 *
 *  The MPI runtime is queried if available for MPI information. After
 *  that, environment variables are checked for common libraries
 *  (SLURM, Open-MPI, MVAPICH2). If neither of these methods gives the
 *  required information, default information consistent with a
 *  sequential job is returned: the rank will be 0 and the size will
 *  be 1.
 *
 *  If the "%env{<NAME>}" substitution fails to find `NAME` in the
 *  current environment, the replacement will be the empty string.
 *
 *  The double-percent sequence is extracted first, so "%%r" will
 *  return "%r" and "%%%r" will return "%<mpi-rank>".
 *
 *  @param str The string to which substitutions should be applied.
 *  @param sys_info The source of system information. This is
 *                  primarily exposed for stubbing the functionality
 *                  to test this function.
 *
 *  @throws BadSubstitutionPattern An escape sequence is found in
 *          the string that has no valid substitution.
 *
 *  @returns A copy of the input string with all substitutions applied.
 */
std::string replace_escapes(
  std::string const& str, lbann::utils::SystemInfo const& sys_info);

/** @brief Indicates that an invalid pattern is detected. */
struct BadSubstitutionPattern : std::runtime_error
{
  BadSubstitutionPattern(std::string const& str);
};// struct BadSubstitutionPattern

}// namespace utilities
}// namespace unit_test
#endif // LBANN_UNIT_TEST_UTILITIES_REPLACE_ESCAPES_HPP_INCLUDED
