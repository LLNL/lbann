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
// patchworks.hpp - LBANN PATCHWORKS main interface header
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS main interface header
 *  - includes the main interface function declarations
 */

#ifdef __LIB_OPENCV
#ifndef _PATCHWORKS_H_INCLUDED_
#define _PATCHWORKS_H_INCLUDED_
#include <vector>
#include "patchworks_common.hpp"
#include "patchworks_patch_descriptor.hpp"

namespace lbann {
namespace patchworks {

/// Compute the min and max value of pixels
std::pair<double,double> check_min_max(const cv::Mat& _img);

/// Adjust for reducing chromatic aberration
cv::Mat correct_chromatic_aberration(const cv::Mat& _img);

/// Drop 2 channels randomly
cv::Mat drop_2channels(const cv::Mat& _img);


/// Take one patch
bool take_patch(const cv::Mat& img, const patch_descriptor& pi, 
                const ROI& roi, std::vector<cv::Mat>& patches);

/// Extract patches according to the given patch description
bool extract_patches(const cv::Mat& img, patch_descriptor& pi, std::vector<cv::Mat>& patches);

} // end of namespace patchworks
} // end of namespace lbann

#endif //_PATCHWORKS_H_INCLUDED_
#endif // __LIB_OPENCV
