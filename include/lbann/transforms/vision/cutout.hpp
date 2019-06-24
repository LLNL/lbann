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

#ifndef LBANN_TRANSFORMS_CUTOUT_HPP_INCLUDED
#define LBANN_TRANSFORMS_CUTOUT_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Cutout data augmentation which randomly masks out square regions of input.
 * 
 * See:
 * 
 *     DeVries and Taylor. "Improved Regularization of Convolutional Neural
 *     Networks with Cutout". arXiv preprint arXiv:1708.04552 (2017).
 *
 * This will randomly select a center pixel for each square and set all pixels
 * within that square to 0. It is permissible for portions of the masks to lie
 * outside of the image.
 *
 * Normalization about 0 should be applied after applying cutout.
 */
class cutout : public transform {
public:
  /**
   * Cutout with a given number of squares of a given size.
   * @param num_holes Number of squares to mask out (must be positive).
   * @param length Length of a side of the square (must be positive).
   */
  cutout(size_t num_holes, size_t length) :
    transform(), m_num_holes(num_holes), m_length(length) {
    if (num_holes == 0) {
      LBANN_ERROR("num_holes must be positive, got 0");
    }
    if (length == 0) {
      LBANN_ERROR("length must be positive, got 0");
    }
  }

  transform* copy() const override { return new cutout(*this); }

  std::string get_type() const override { return "cutout"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;

private:
  /** Number of squares that will be masked out. */
  size_t m_num_holes;
  /** Length of a side of each square that will be masked out. */
  size_t m_length;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_CUTOUT_HPP_INCLUDED
