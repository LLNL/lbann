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

#ifndef LBANN_UTILS_SYSTEM_INFO_HPP_INCLUDED
#define LBANN_UTILS_SYSTEM_INFO_HPP_INCLUDED

#include <string>

namespace lbann {
namespace utils {

/** @class SystemInfo
 *  @brief Query basic system information
 *
 *  The class structure here is, strictly speaking, unnecessary. It is
 *  used to provide a "hook" for stubbing this information during
 *  testing.
 */
class SystemInfo
{
public:
  /** @brief Virtual destructor */
  virtual ~SystemInfo() noexcept = default;

  /** @brief Get the current process ID.
   *
   *  This returns the value as a string to avoid system differences
   *  in `pid_t`. However, it's probably safe to return either int64_t
   *  or uint64_t here.
   */
  virtual std::string pid() const;

  /** @brief Get the host name for this process. */
  virtual std::string host_name() const;

  /** @brief Get the MPI rank of this process.
   *
   *  If this is not an MPI job, or cannot be determined to be an MPI
   *  job, this will return 0.
   *
   *  The return type is chosen for consistency with MPI 3.0.
   */
  virtual int mpi_rank() const;

  /** @brief Get the size of the MPI universe in which this process is
   *         participating.
   *
   *  If this is not an MPI job, or cannot be determined to be an MPI
   *  job, this will return 1.
   *
   *  The return type is chosen for consistency with MPI 3.0.
   */
  virtual int mpi_size() const;

  /** @brief Get the value of the given variable from the environment.
   *
   *  If the variable doesn't exist, the empty string is returned.
   */
  virtual std::string env_variable_value(std::string const& var_name) const;

};

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_SYSTEM_INFO_HPP_INCLUDED
