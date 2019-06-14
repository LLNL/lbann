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

#include "lbann/data_readers/cv_decolorizer.hpp"
#include "lbann/utils/mild_exception.hpp"

#ifdef LBANN_HAS_OPENCV
namespace lbann {

cv_decolorizer::cv_decolorizer(const cv_decolorizer& rhs)
  : cv_transform(rhs), m_color(rhs.m_color), m_pick_1ch(rhs.m_pick_1ch) {}

cv_decolorizer& cv_decolorizer::operator=(const cv_decolorizer& rhs) {
  cv_transform::operator=(rhs);
  m_color = rhs.m_color;
  m_pick_1ch = rhs.m_pick_1ch;
  return *this;
}

cv_decolorizer *cv_decolorizer::clone() const {
  return (new cv_decolorizer(*this));
}

void cv_decolorizer::set(const bool pick_1ch) {
  m_pick_1ch = pick_1ch;
  reset();
}

bool cv_decolorizer::determine_transform(const cv::Mat& image) {
  //reset(); // redundant here
  // enable decolorizing transform if the given image is a color image
  m_enabled = m_color = (!image.empty() && (image.channels() > 1));
  //_LBANN_SILENT_EXCEPTION(image.empty(), "", false); // redundant
  return m_enabled;
}

bool cv_decolorizer::apply(cv::Mat& image) {
  m_enabled = false; // turn off as the transform is applied once

  if (m_color) {
    if (m_pick_1ch) {
      // Drop all the channels but one.
      const int Nch = image.channels();
      std::vector<cv::Mat> channels(Nch);
      cv::split(image, channels);
      image = channels[1 % Nch];
    } else {
      // Compute a new channel by the linear combination of all channels
      cv::Mat image_dst;
      cv::cvtColor(image, image_dst, cv::COLOR_BGR2GRAY);
      image = image_dst;
    }
  }

  return true;
}

std::string cv_decolorizer::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl;
  return os.str();
}

std::ostream& cv_decolorizer::print(std::ostream& os) const {
  os << get_description()
     << " - " << (m_color? "color" : "grayscale") << std::endl;
  return os;
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
