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
#ifndef LBANN_UTILS_DIM_HELPERS_HPP_
#define LBANN_UTILS_DIM_HELPERS_HPP_

#include <lbann/utils/exception.hpp>

namespace lbann
{

template <typename T>
auto get_linear_size(std::vector<T> const& dims)
{
  return (dims.size()
          ? std::accumulate(begin(dims), end(dims), T(1), std::multiplies<T>())
          : T(0));
}

template <typename T>
auto get_linear_size(size_t ndims, T const* dims)
{
  return (ndims
          ? std::accumulate(dims, dims+ndims, T(1), std::multiplies<T>())
          : T(0));
}


template <typename T>
auto get_strides(size_t ndims, T const* dims, T const& lowest_stride)
{
  std::vector<T> strides(ndims, lowest_stride);
  if (ndims > 0)
  {
    for (size_t ii = ndims-1; ii != 0; --ii)
    {
      if (dims[ii] == T(0))
        LBANN_ERROR("Zero-sized dimension not allowed. Dims[",ii,"] = 0.");
      strides[ii-1] = strides[ii]*dims[ii];
    }
  }
  return strides;
}

template <typename T>
auto get_strides(std::vector<T> const& dims, T const& lowest_stride)
{
  return get_strides(dims.size(), dims.data(), lowest_stride);
}

template <typename T>
auto get_packed_strides(size_t ndims, T const* dims)
{
  return get_strides(ndims, dims, T(1));
}

template <typename T>
auto get_packed_strides(std::vector<T> const& dims)
{
  return get_strides(dims.size(), dims.data(), T(1));
}

}// namespace lbann
#endif // LBANN_UTILS_DIM_HELPERS_HPP_
