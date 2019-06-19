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

#include "lbann/transforms/transform_pipeline.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {
namespace transform {

transform_pipeline::transform_pipeline(const transform_pipeline& other) :
  m_expected_out_dims(other.m_expected_out_dims) {
  for (const auto& trans : other.m_transforms) {
    m_transforms.emplace_back(trans->copy());
  }
}

transform_pipeline& transform_pipeline::operator=(
  const transform_pipeline& other) {
  m_expected_out_dims = other.m_expected_out_dims;
  m_transforms.clear();
  for (const auto& trans : other.m_transforms) {
    m_transforms.emplace_back(trans->copy());
  }
  return *this;
}

void transform_pipeline::apply(utils::type_erased_matrix& data,
                               std::vector<size_t>& dims) {
  for (auto& trans : m_transforms) {
    trans->apply(data, dims);
  }
  assert_expected_out_dims(dims);
}

void transform_pipeline::apply(CPUMat& data, std::vector<size_t>& dims) {
  utils::type_erased_matrix m = utils::type_erased_matrix(std::move(data));
  apply(m, dims);
  data = std::move(m.template get<DataType>());
}

void transform_pipeline::apply(El::Matrix<uint8_t>& data, CPUMat& out_data,
                               std::vector<size_t>& dims) {
  utils::type_erased_matrix m = utils::type_erased_matrix(std::move(data));
  if (!m_transforms.empty()) {
    bool applied_non_inplace = false;
    size_t i = 0;
    for (; !applied_non_inplace && i < m_transforms.size(); ++i) {
      if (m_transforms[i]->supports_non_inplace()) {
        applied_non_inplace = true;
        m_transforms[i]->apply(m, out_data, dims);
      } else {
        m_transforms[i]->apply(m, dims);
      }
    }
    if (!applied_non_inplace) {
      LBANN_ERROR("No transform to go from uint8 -> DataType");
    }
    if (i < m_transforms.size()) {
      // Apply the remaining transforms.
      // TODO(pp): Prevent out_data from being resized/reallocated.
      m = utils::type_erased_matrix(std::move(out_data));
      for (; i < m_transforms.size(); ++i) {
        m_transforms[i]->apply(m, dims);
      }
      out_data = std::move(m.template get<DataType>());
    }
  } else {
    LBANN_ERROR("No transform to go from uint8 -> DataType");
  }
  assert_expected_out_dims(dims);
}

void transform_pipeline::assert_expected_out_dims(
  const std::vector<size_t>& dims) {
  if (!m_expected_out_dims.empty() && dims != m_expected_out_dims) {
    std::stringstream ss;
    ss << "Transformed dims do not match expected dims, got {";
    for (const auto& d : dims) { ss << d << " "; }
    ss << "} expected {";
    for (const auto& d : m_expected_out_dims) { ss << d << " "; }
    ss << "}";
    LBANN_ERROR(ss.str());
  }
}

}  // namespace transform
}  // namespace lbann
