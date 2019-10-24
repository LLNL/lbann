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

#ifndef LBANN_UTILS_ENVIRONMENT_VARIABLE_HPP_INCLUDED
#define LBANN_UTILS_ENVIRONMENT_VARIABLE_HPP_INCLUDED

#include "lbann/utils/from_string.hpp"

#include <string>

namespace lbann
{
namespace utils
{

/** @brief Access environment variables using getenv. */
class GetEnvAccessor
{
public:
  std::string get(std::string const& var_name) const;
};

/** @brief An environment variable
 *
 *  Values are acquired lazily. The only maintained state is the name.
 */
template <typename AccessPolicy=GetEnvAccessor>
class EnvVariable
{
public:

  /** @name Constructors */
  ///@{

  /** @brief Construct from a string. */
  EnvVariable(std::string const& var_name);

  /** @brief Construct from a temporary string. */
  EnvVariable(std::string&& var_name);

  ///@}
  /** @name Queries */
  ///@{

  /** @brief Test if the variable exists in the environment.
   *
   *  Existence means set to a nonempty string.
   */
  bool exists() const;

  ///@}
  /** @name Accessors */
  ///@{

  /** @brief Get the name of the environment variable. */
  std::string const& name() const noexcept;

  /** @brief Get the string value of the environment variable. */
  std::string raw_value() const;

  /** @brief Get the value of the environment variable as a certain type. */
  template <typename T>
  T value() const;

  ///@}

private:
  /** @brief The name of the variable. */
  std::string name_;
};

// Implementation

template <typename AccessPolicy>
inline
EnvVariable<AccessPolicy>::
EnvVariable(std::string const& var_name)
  : name_{var_name}
{}

template <typename AccessPolicy>
inline
EnvVariable<AccessPolicy>::
EnvVariable(std::string&& var_name)
  : name_{std::move(var_name)}
{}

template <typename AccessPolicy>
inline bool
EnvVariable<AccessPolicy>::exists() const
{
  return raw_value().size() > 0;
}

template <typename AccessPolicy>
inline std::string const&
EnvVariable<AccessPolicy>::name() const noexcept
{
  return name_;
}

template <typename AccessPolicy>
inline std::string
EnvVariable<AccessPolicy>::raw_value() const
{
  AccessPolicy access;
  return access.get(name_);
}

template <typename AccessPolicy>
template <typename T>
T EnvVariable<AccessPolicy>::value() const
{
  return from_string<T>(raw_value());
}

}// namespace utils
}// namespace lbann
#endif /* LBANN_UTILS_ENVIRONMENT_VARIABLE_HPP_INCLUDED */
