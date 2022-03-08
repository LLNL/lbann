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

#ifndef LBANN_UTILS_TYPENAME_HPP_INCLUDED
#define LBANN_UTILS_TYPENAME_HPP_INCLUDED

#include <lbann_config.hpp>

#include "lbann/base.hpp"

#include <complex>
#include <string>
#include <typeinfo>

namespace lbann {
namespace details {
std::string get_type_name(std::type_info const&);
}

template <typename T>
std::string TypeName()
{
  return details::get_type_name(typeid(T));
}

#define ADD_TYPENAME_INST(Type)                                     \
  template <> inline std::string TypeName<Type>() { return #Type; }

ADD_TYPENAME_INST(float)
ADD_TYPENAME_INST(double)
#ifdef LBANN_HAS_HALF
ADD_TYPENAME_INST(cpu_fp16)
#endif
#ifdef LBANN_HAS_GPU_FP16
ADD_TYPENAME_INST(fp16)
#endif
ADD_TYPENAME_INST(std::complex<float>)
ADD_TYPENAME_INST(std::complex<double>)

} // namespace lbann

#endif // LBANN_UTILS_TYPENAME_HPP_INCLUDED
