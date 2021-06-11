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

#ifndef LBANN_VECTOR_DYNAMIC_CAST_HPP_INCLUDED
#define LBANN_VECTOR_DYNAMIC_CAST_HPP_INCLUDED

namespace lbann {

template <typename OutType, typename InType>
auto vector_dynamic_cast(std::vector<InType*>const& v_in) -> std::vector<OutType*>
{
  static_assert(std::is_base_of_v<InType, OutType>,
                "InType must be base of OutType");
  std::vector<OutType*> v;
  v.reserve(v_in.size());
  for (auto& e : v_in) {
#ifdef LBANN_DEBUG
    auto* ptr = dynamic_cast<OutType*>(e);
    LBANN_ASSERT(ptr);
    v.push_back(ptr);
#else
    v.push_back(dynamic_cast<OutType*>(e));
#endif
  }
  return v;
}

} // lbann

#endif // LBANN_VECTOR_DYNAMIC_CAST_HPP_INCLUDED
