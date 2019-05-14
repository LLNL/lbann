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

#ifndef LBANN_TRANSFORMS_TRANSFORM_PIPELINE_HPP_INCLUDED
#define LBANN_TRANSFORMS_TRANSFORM_PIPELINE_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Applies a sequence of transforms to input data.
 */
class transform_pipeline {
public:
  transform_pipeline() {}
  transform_pipeline(const transform_pipeline&);
  transform_pipeline(transform_pipeline&&) = default;
  transform_pipeline& operator=(const transform_pipeline&);
  transform_pipeline& operator=(transform_pipeline&&) = default;
  ~transform_pipeline() {}

  transform_pipeline* copy() const { return new transform_pipeline(*this); }

  /**
   * Add trans as the next transform to apply.
   */
  void add_transform(std::unique_ptr<transform> trans) {
    m_transforms.push_back(std::move(trans));
  }

  /**
   * Set the expected dimensions of the data after applying the transforms.
   * This is primarily meant as a debugging aid/sanity check.
   */
  void set_expected_out_dims(std::vector<size_t> expected_out_dims) {
    m_expected_out_dims = expected_out_dims;
  }

  /**
   * Apply the transforms to data.
   * @param data The data to transform. data will be modified in-place.
   * @param dims Dimensions of data. Will be modified in-place.
   */
  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims);
  /** Apply to CPUMat data, which will be modified in-place. */
  void apply(CPUMat& data, std::vector<size_t>& dims);
  /**
   * Apply the transforms to data.
   * @param data The data to transform. Will be modified in-place.
   * @param out_data Output will be placed here. It will not be reallocated.
   * @param dims Dimensions of data. Will be modified in-place.
   */
  void apply(El::Matrix<uint8_t>& data, CPUMat& out_data,
             std::vector<size_t>& dims);
private:
  /** Ordered list of transforms to apply. */
  std::vector<std::unique_ptr<transform>> m_transforms;
  /** Expected dimensions after applying all transforms. */
  std::vector<size_t> m_expected_out_dims;

  /** Assert dims matches expected_out_dims (if set). */
  void assert_expected_out_dims(const std::vector<size_t>& dims);
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_TRANSFORM_PIPELINE_HPP_INCLUDED
