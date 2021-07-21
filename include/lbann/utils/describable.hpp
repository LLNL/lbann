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
#ifndef LBANN_UTILS_DESCRIBABLE_HPP_INCLUDED
#define LBANN_UTILS_DESCRIBABLE_HPP_INCLUDED

#include "lbann/utils/description.hpp"

namespace lbann {

/** @brief Non-intrusive capitalization fix. */
using Description = description;

/** @brief A class that can generate self-descriptions. */
class Describable
{
public:
  virtual ~Describable() = default;
  /** @brief Generate a human-readable description. */
  virtual Description get_description() const = 0;
}; // class Describable

inline std::ostream& operator<<(std::ostream& os, Describable const& obj)
{
  return os << obj.get_description();
}

} // namespace lbann
#endif // LBANN_UTILS_DESCRIBABLE_HPP_INCLUDED
