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
//
// cv_decolorizer .cpp .hpp - transform a color image into a single-channel
//                            monochrome image
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_DECOLORIZER_HPP
#define LBANN_CV_DECOLORIZER_HPP

#include "lbann_config.hpp"
#include "cv_transform.hpp"

#ifdef LBANN_HAS_OPENCV
namespace lbann {

class cv_decolorizer : public cv_transform {
 protected:
  // --- state variables ---
  bool m_color; ///< whether an image is color or not
  /// Method to used: either pick one channel, or mix BGR channels (default)
  bool m_pick_1ch;

 public:
  cv_decolorizer() : cv_transform(), m_color(false), m_pick_1ch(false) {}
  cv_decolorizer(const cv_decolorizer& rhs);
  cv_decolorizer& operator=(const cv_decolorizer& rhs);
  cv_decolorizer *clone() const override;

  ~cv_decolorizer() override {}

  void set(const bool pick_1ch);
  void reset() override {
    m_enabled = false;
    m_color = false;
  }

  /**
   * If a given image is in color, the tranform is enabled, and not otherwise.
   * @return false if not enabled or unsuccessful.
   */
  bool determine_transform(const cv::Mat& image) override;

  /// The decolorizing transform is irreversible. Thus, this has no effect.
  bool determine_inverse_transform() override { return false; }

  /**
   * Convert a color image to a monochrome image if enabled.
   * As it is applied, the transform becomes deactivated.
   * @return false if not successful.
   */
  bool apply(cv::Mat& image) override;

  std::string get_type() const override { return "decolorizer"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_DECOLORIZER_HPP
