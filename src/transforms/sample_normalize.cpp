////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/transforms/sample_normalize.hpp"
#include "lbann/utils/statistics.hpp"

namespace lbann {
namespace transform {

void sample_normalize::apply(utils::type_erased_matrix& data, std::vector<size_t>&) {
  // Only work with DataTypes to avoid rounding/floating point issues.
  auto& mat = data.template get<DataType>();
  if (mat.Height() != mat.LDim()) {
    LBANN_ERROR("Normalizing non-contiguous matrix not supported.");
  }
  DataType mean, stdev;
  entrywise_mean_and_stdev(mat, mean, stdev);
  DataType* __restrict__ buf = mat.Buffer();
  const El::Int size = mat.Height() * mat.Width();
  for (El::Int i = 0; i < size; ++i) {
    buf[i] = (buf[i] - mean) / stdev;
  }
}

}  // namespace transform
}  // namespace lbann
