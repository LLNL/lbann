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

#ifndef LBANN_TRANSFORMS_ADJUST_BRIGHTNESS_HPP_INCLUDED
#define LBANN_TRANSFORMS_ADJUST_BRIGHTNESS_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/** Adjust the brightness of an image. */
class adjust_brightness : public transform {
public:
  /**
   * Adjust brightness with given factor.
   * @param factor A non-negative factor. 0 gives a black image, 1 the original.
   */
  adjust_brightness(float factor) : transform(), m_factor(factor) {
    if (factor < 0.0f) {
      LBANN_ERROR("Brightness factor must be non-negative.");
    }
  }
  
  transform* copy() const override { return new adjust_brightness(*this); }

  std::string get_type() const override { return "adjust_brightness"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;

private:
  /** Factor to adjust brightness by. */
  float m_factor;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_ADJUST_BRIGHTNESS_HPP_INCLUDED
