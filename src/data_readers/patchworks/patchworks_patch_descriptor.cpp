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
//
// patchworks_patch_descriptor.cpp - LBANN PATCHWORKS implementation for patch descriptor
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS implementation for patch descriptor
 */

#include <iostream>
#include "lbann/data_readers/patchworks/patchworks_patch_descriptor.hpp"

namespace lbann {
namespace patchworks {

void patch_descriptor::init(const int seed)
{
  if (seed == 0)
    m_rg_jitter.reset();
  else
    m_rg_jitter.reset(static_cast<rand_t::generator_t::result_type>(seed));

  m_width = 0u;
  m_height = 0u;
  m_gap = 0u;
  m_jitter = 0u;
  m_mode_center = 1u;
  m_mode_chrom = 0u;
  m_ext = "";
  m_cur_patch_idx = 0u;
}

void patch_descriptor::set_seed(const int seed)
{
  m_rg_jitter.reset(static_cast<rand_t::generator_t::result_type>(seed));
}

void patch_descriptor::set_size(const int width, const int height)
{
  m_width = width;
  m_height = height;
}

bool patch_descriptor::set_sample_area(const ROI& area)
{
  if (!area.is_valid()) return false;
  m_sample_area = area;
  return true;
}

bool patch_descriptor::set_sample_image(const unsigned int img_width, const unsigned int img_height)
{
  ROI whole_image;
  whole_image.set_by_corners(0, 0, img_width, img_height);

  return set_sample_area(whole_image);
}

void patch_descriptor::define_patch_set(void)
{
  const int wdisp = m_width + m_gap;
  const int hdisp = m_height + m_gap;
  m_displacements.clear();
  m_displacements.push_back(std::make_pair(-wdisp, -hdisp));
  m_displacements.push_back(std::make_pair( 0,     -hdisp));
  m_displacements.push_back(std::make_pair( wdisp, -hdisp));
  m_displacements.push_back(std::make_pair(-wdisp,  0));
  m_displacements.push_back(std::make_pair( wdisp,  0));
  m_displacements.push_back(std::make_pair(-wdisp,  hdisp));
  m_displacements.push_back(std::make_pair( 0,      hdisp));
  m_displacements.push_back(std::make_pair( wdisp,  hdisp));
}

void patch_descriptor::set_jitter(const unsigned int j)
{
  m_jitter = j;
  m_rg_jitter.init_uniform_int(0, 0, 2*m_jitter);
  m_rg_jitter.init_uniform_int(1, 0, 2*m_jitter);
}

bool patch_descriptor::get_first_patch(ROI& patch)
{
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
    if (!center_patch_area.is_valid()) return false;

    // randomly generate the center coordinate within the center patch area
    rand_t rg_center;
    rg_center.init_uniform_int(0, 0, center_patch_area.width()-1);
    rg_center.init_uniform_int(1, 0, center_patch_area.height()-1);

    x_center = rg_center.gen_uniform_int(0) + center_patch_area.left();
    y_center = rg_center.gen_uniform_int(1) + center_patch_area.top();
  }

  if (m_jitter > 0u) { // apply position jitter if enabled
    x_center += m_rg_jitter.gen_uniform_int(0) - m_jitter;
    y_center += m_rg_jitter.gen_uniform_int(1) - m_jitter;
  }

  // set the center patch
  ROI p;
  if (!p.set_by_center(x_center, y_center, m_width, m_height) ||
      !(m_sample_area >= p))
    return false;

  m_patch_center = p;
  patch = p;
  positions.clear();
  positions.push_back(patch);

  return true;
}

bool patch_descriptor::get_next_patch(ROI& patch)
{
  bool got_one = false;

  do {
    ROI p = m_patch_center;

    if (m_cur_patch_idx >= m_displacements.size()) return false;
    p.move(m_displacements[m_cur_patch_idx++]);

    if (m_jitter > 0u) {
      const int x_jitter = m_rg_jitter.gen_uniform_int(0) - m_jitter;
      const int y_jitter = m_rg_jitter.gen_uniform_int(1) - m_jitter;
      p.move(displacement_type(x_jitter, y_jitter));
    }

    if (p.is_valid() && (m_sample_area >= p)) {
        patch = p;
        got_one = true;
    }
  } while (!got_one);

  positions.push_back(patch);
  return true;
}

std::ostream& patch_descriptor::print(std::ostream& os) const
{
  os << " m_width: " << m_width << std::endl
     << " m_height: " << m_height << std::endl
     << " m_gap: " << m_gap << std::endl
     << " m_jitter: " << m_jitter << std::endl
     << " m_mode_center: " << m_mode_center << std::endl
     << " m_mode_chrom: " << m_mode_chrom << std::endl
     << " m_ext: " << m_ext << std::endl
     << " m_sample_area: " << m_sample_area << std::endl
     << " m_cur_patch_idx: " << m_cur_patch_idx << std::endl;

  os << "patch displacements from the center: " << std::endl;
  for (unsigned int i=0u; i < m_displacements.size() ; ++i) {
    os << i+1 << ' ' << m_displacements[i].first << ' ' << m_displacements[i].second << std::endl;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const patch_descriptor& pd) { return pd.print(os); }

} // end of namespace patchworks
} // end of namespace lbann
