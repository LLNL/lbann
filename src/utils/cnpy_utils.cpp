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

#include "lbann/utils/cnpy_utils.hpp"

namespace lbann {
namespace cnpy_utils {

size_t compute_cnpy_array_offset(
  const cnpy::NpyArray& na, std::vector<size_t> indices) {

  if (indices.size() < na.shape.size()) {
    indices.resize(na.shape.size(), 0u);
  }
  bool ok = (indices.size() == na.shape.size());
  size_t unit_stride = 1u;
  size_t offset = 0u;

  for (size_t i = indices.size(); ok && (i-- > 0u); ) {
    ok = (indices[i] < na.shape[i]) ||
        // relax to allow representing the exclusive upper bound
        ((i == 0u) && (indices[i] == na.shape[i]));
    offset += indices[i] * unit_stride;
    unit_stride *= na.shape[i];
  }
  if (!ok) {
    throw lbann_exception("compute_cnpy_array_offset(): invalid data index");
  }
  return offset;
}


void shrink_to_fit(cnpy::NpyArray& na, size_t sz) {
  if ((na.shape.size() == 0u) || (na.shape[0] <= sz)) {
    //std::cerr << "not able to shrink to " << sz << std::endl;
    return;
  }
  size_t new_size = sz;
  for(size_t i = 1u; i < na.shape.size(); ++i) {
    new_size *= na.shape[i];
  }
  na.data_holder->resize(new_size*na.word_size);
  na.data_holder->shrink_to_fit();
  na.num_vals = new_size;
  na.shape[0] = sz;
}


std::string show_shape(const cnpy::NpyArray& na) {
  std::string ret;
  for (const size_t s: na.shape) {
    ret += std::to_string(s) + 'x';
  }
  if (ret.size() == 0u) {
    return "empty";
  } else {
    ret.pop_back(); // remove the last 'x'
    ret += " " + std::to_string(na.word_size);
  }
  return ret;
}

} // end of namespace cnpy_utils
} // end of namespace lbann
