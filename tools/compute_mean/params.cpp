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
////////////////////////////////////////////////////////////////////////////////

#include "params.hpp"
#include <sstream>


namespace tools_compute_mean {

/// Parse the command line arguments.
bool params::set(int argc, char *argv[]) {
  if (argc == 2) {
    m_data_path_file = argv[1];
    m_only_create_output_dirs = true;
    m_write_cropped = true;
    return true;
  } else if ((argc != 6) && (argc != 9)) {
    return false;
  }
  m_only_create_output_dirs = false;
  unsigned int aidx = 0u;

  m_data_path_file = argv[++aidx];
  m_enable_cropper = true;

  cp.m_is_set = true;
  cp.m_crop_sz.first = atoi(argv[++aidx]);
  cp.m_crop_sz.second = atoi(argv[++aidx]);

  if (argc == 6) {
    cp.m_rand_center = false;
    cp.m_roi_sz.first = cp.m_crop_sz.first;
    cp.m_roi_sz.second = cp.m_crop_sz.second;
  } else {
    cp.m_rand_center = static_cast<bool>(atoi(argv[++aidx]));
    cp.m_roi_sz.first = atoi(argv[++aidx]);
    cp.m_roi_sz.second = atoi(argv[++aidx]);
  }

  m_mean_batch_size = atoi(argv[++aidx]);
  m_write_cropped = static_cast<bool>(atoi(argv[++aidx])) && m_enable_cropper;
  m_enable_mean_extractor = (m_mean_batch_size > 0);

  return true;
}


/// Print out the list of command line arguments used.
std::string params::show_help(std::string name) {
  std::ostringstream oss;

  oss << "Usage: > " << name << " path_file w h [ r rw rh ] bsz save" << std::endl
      << "       > " << name << " path_file" << std::endl
      << std::endl
      << "  path_file: contains the paths to the root data directory, the image list file" << std::endl
      << "             and the output directory. The list file contains the relative path" << std::endl
      << "             of each image to the root directory." << std::endl
      << "             If this is the only argument, it will create the output directories and exit." << std::endl
      << "  The parameters w, h, c, rw and rh are used by cropper." << std::endl
      << "    w: the final crop width of image" << std::endl
      << "    h: the final crop height of image." << std::endl
      << "       (w and h are dictated whether by cropping images to the size)" << std::endl
      << "    r: whether to randomize the crop position within the center region. (0|1)" << std::endl
      << "   rw: The width of the center region with respect to w after resizig the raw image." << std::endl
      << "   rh: The height of the center region with respect to h after resizing the raw image." << std::endl
      << "       Raw image will be resized to an image of size rw x rh around the center," << std::endl
      << "       which covers area of the original image as much as possible while preseving" << std::endl
      << "       the aspect ratio of object in the image." << std::endl
      << "       When r, rw, and rh are omitted, it is assumed that r=0, rw=w, and rh=h." << std::endl
      << std::endl
      << "  bsz: The batch size for mean extractor." << std::endl
      << "       if 0, turns off the mean extractor." << std::endl
      << "  save: write cropped images. (0|1)" << std::endl
      << std::endl;

  return oss.str();
}

} // end of namespace tools_compute_mean
