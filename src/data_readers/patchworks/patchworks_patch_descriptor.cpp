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
// patchworks_patch_descriptor.cpp - LBANN PATCHWORKS implementation for patch descriptor
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS implementation for patch descriptor
 */

#include "lbann/data_readers/patchworks/patchworks_patch_descriptor.hpp"

#ifdef LBANN_HAS_OPENCV
#include <iostream>
#include "lbann/utils/random.hpp"

namespace lbann {
namespace patchworks {

void patch_descriptor::init() {
  m_width = 0u;
  m_height = 0u;
  m_gap = 0u;
  m_jitter = 0u;
  m_mode_center = 1u;
  m_mode_chrom = 0u;
  m_self_label = false;
  m_ext = "";
  m_sample_area = ROI();
  m_displacements.clear();
  reset();
}

void patch_descriptor::reset() {
  m_patch_center = ROI();
  m_positions.clear();
  m_cur_patch_idx = 0u;
}

void patch_descriptor::set_size(const int width, const int height) {
  m_width = width;
  m_height = height;
}

bool patch_descriptor::set_sample_area(const ROI& area) {
  if (!area.is_valid()) {
    return false;
  }
  m_sample_area = area;
  return true;
}

bool patch_descriptor::set_sample_image(const unsigned int img_width, const unsigned int img_height) {
  ROI whole_image;
  whole_image.set_by_corners(0, 0, img_width, img_height);

  return set_sample_area(whole_image);
}

void patch_descriptor::define_patch_set() {
  const int wdisp = m_width + m_gap;
  const int hdisp = m_height + m_gap;
  m_displacements.clear();
  m_displacements.emplace_back(-wdisp, -hdisp);
  m_displacements.emplace_back( 0,     -hdisp);
  m_displacements.emplace_back( wdisp, -hdisp);
  m_displacements.emplace_back(-wdisp,  0);
  m_displacements.emplace_back( wdisp,  0);
  m_displacements.emplace_back(-wdisp,  hdisp);
  m_displacements.emplace_back( 0,      hdisp);
  m_displacements.emplace_back( wdisp,  hdisp);
}

bool patch_descriptor::get_first_patch(ROI& patch) {
  int x_center = (m_sample_area.width()+1)/2 + m_sample_area.left();
  int y_center = (m_sample_area.height()+1)/2 + m_sample_area.top();
  int x_margin = 0;
  int y_margin = 0;

  if (m_mode_center == 0u) {
    // Consider the jitter for a patch at the boundary of an image
    x_margin = (m_width+1)/2 + m_jitter;
    y_margin = (m_height+1)/2 + m_jitter;
  } else if (m_mode_center == 1u) {
    // The jitter for the center patch is a part of gap.
    //if (m_jitter > m_gap) return false;
    x_margin = m_width + (m_width+1)/2 + 2*m_jitter + m_gap;
    y_margin = m_height + (m_height+1)/2 + 2*m_jitter + m_gap;
  }

  ::lbann::rng_gen& gen = ::lbann::get_io_generator();

  if ((m_mode_center == 0u || m_mode_center == 1u)) {
    // area where the center of a center patch can be in
    ROI center_patch_area;
    bool ok = center_patch_area.set_by_corners(x_margin + m_sample_area.left(),
              y_margin + m_sample_area.top(),
              m_sample_area.width() - x_margin,
              m_sample_area.height() - y_margin);
    if (!ok) {
      std::cout << "invalid center patch area: " << center_patch_area << std::endl;
      return false;
    }
    if (!center_patch_area.is_valid()) {
      return false;
    }

    // randomly generate the center coordinate within the center patch area
    std::uniform_int_distribution<int> rg_center_x(0, center_patch_area.width()-1);
    std::uniform_int_distribution<int> rg_center_y(0, center_patch_area.height()-1);
    x_center = rg_center_x(gen) + center_patch_area.left();
    y_center = rg_center_y(gen) + center_patch_area.top();
  }

  if (m_jitter > 0u) { // apply position jitter if enabled
    std::uniform_int_distribution<int> rg_jitter_x(0, 2*m_jitter);
    std::uniform_int_distribution<int> rg_jitter_y(0, 2*m_jitter);
    x_center += rg_jitter_x(gen) - m_jitter;
    y_center += rg_jitter_y(gen) - m_jitter;
  }

  // set the center patch
  ROI p;
  if (!p.set_by_center(x_center, y_center, m_width, m_height) ||
      !(m_sample_area >= p)) {
    return false;
  }

  m_patch_center = p;
  patch = p;
  m_positions.clear();
  m_cur_patch_idx = 0u;
  m_positions.push_back(patch);

  return true;
}

bool patch_descriptor::get_next_patch(ROI& patch) {
  bool got_one = false;

  ::lbann::rng_gen& gen = ::lbann::get_io_generator();

  do {
    ROI p = m_patch_center;

    if (m_cur_patch_idx >= m_displacements.size()) {
      return false;
    }
    p.move(m_displacements[m_cur_patch_idx++]);

    if (m_jitter > 0u) {
      std::uniform_int_distribution<int> rg_jitter_x(0, 2*m_jitter);
      std::uniform_int_distribution<int> rg_jitter_y(0, 2*m_jitter);
      const int x_jitter = rg_jitter_x(gen) - m_jitter;
      const int y_jitter = rg_jitter_y(gen) - m_jitter;
      p.move(displacement_type(x_jitter, y_jitter));
    }

    if (p.is_valid() && (m_sample_area >= p)) {
      patch = p;
      got_one = true;
    }
  } while (!got_one);

  m_positions.push_back(patch);
  return true;
}

bool patch_descriptor::extract_patches(const cv::Mat& img, std::vector<cv::Mat>& patches) {
  patches.clear();
  if (img.data == nullptr) {
    return false;
  }

  ROI roi;
  bool ok = get_first_patch(roi);
  if (!ok) {
    return false;
  }

  patches.push_back(img(roi.rect()).clone());

#if 0 // to generate all the patches defined in the set
  unsigned int i = 1u;

  while (get_next_patch(roi)) {
    patches.push_back(img(roi.rect()).clone());
    i++;
  }
  if (i == 1u) {
    return false;
  }
#else // to randomly generate another patch. The label will be recorded to m_cur_patch_idx.
  if (m_displacements.size() == 0) {
    return false;
  }

  std::uniform_int_distribution<int> rg_patch_idx(0, m_displacements.size()-1);
  ::lbann::rng_gen& gen = ::lbann::get_io_generator();
  m_cur_patch_idx = rg_patch_idx(gen);

  if (!get_next_patch(roi)) {
    return false;
  }
  patches.push_back(img(roi.rect()).clone());
#endif

  return true;
}

std::string patch_descriptor::get_description() const {
  std::stringstream os;
  os << "patch descriptor:" << std::endl
     << '\t' << "m_width: " << m_width << std::endl
     << '\t' << "m_height: " << m_height << std::endl
     << '\t' << "m_gap: " << m_gap << std::endl
     << '\t' << "m_jitter: " << m_jitter << std::endl
     << '\t' << "m_mode_center: " << m_mode_center << std::endl
     << '\t' << "m_mode_chrom: " << m_mode_chrom << std::endl
     << '\t' << "m_self_label: " << m_self_label << std::endl
     << '\t' << "m_ext: " << m_ext << std::endl
     << '\t' << "m_sample_area: " << m_sample_area << std::endl
     << '\t' << "patch displacements from the center: " << std::endl;
  for (unsigned int i=0u; i < m_displacements.size() ; ++i) {
    os << "\t\t" << i+1 << ' ' << m_displacements[i].first << ' ' << m_displacements[i].second << std::endl;
  }

  return os.str();
}

std::ostream& patch_descriptor::print(std::ostream& os) const {
  os << get_description()
     << '\t' << "m_cur_patch_idx: " << m_cur_patch_idx << std::endl
     << '\t' << "patch regions: " << std::endl;
  for (unsigned int i=0u; i < m_positions.size() ; ++i) {
    os << "\t\t" << i << '\t' << m_positions[i] << std::endl;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const patch_descriptor& pd) {
  return pd.print(os);
}

} // end of namespace patchworks
} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
