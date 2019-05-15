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
// patchworks_stats.hpp - LBANN PATCHWORKS header for pixel statistics
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS header for pixel statistics
 */

#include "lbann_config.hpp"

#ifdef LBANN_HAS_OPENCV
#ifndef _PATCHWORKS_STATS_INCLUDED_
#define _PATCHWORKS_STATS_INCLUDED_

#include <iostream>
#include <vector>
#include "patchworks_common.hpp"

namespace lbann {
namespace patchworks {

/// Pixel statistics of an image
struct image_stats {
  size_t cnt; ///< number of values (pixels)
  size_t cntZeros; ///< number of zero values
  pw_fp_t min; ///< minimum intensity of a pixel
  pw_fp_t max; ///< maximum intensity of a pixel
  pw_fp_t median; ///< median intensity of a pixel
  pw_fp_t minNZ; ///< number of non-zero pixels
  pw_fp_t medianNZ; ///< median among non-zero values
  double avg; ///< average intensity
  double avgNZ; ///< average intensity among non-zeros
  double stdev; ///< standard deviation of intensity
  double stdevNZ; ///< standard deviation among non-zero values

  /// Print out statistics
  std::ostream& Print(std::ostream& os) const {
    os << "   stats:" << std::endl
       << "    - cnt   : " << cnt << std::endl
       << "    - cnt0  : " << cntZeros << std::endl
       << "    - min   : " << min << std::endl
       << "    - max   : " << max << std::endl
       << "    - med   : " << median << std::endl
       << "    - minNZ : " << minNZ << std::endl
       << "    - medNZ : " << medianNZ << std::endl
       << "    - avg   : " << avg << std::endl
       << "    - avgNZ : " << avgNZ << std::endl
       << "    - std   : " << stdev << std::endl
       << "    - stdNZ : " << stdevNZ << std::endl;
    return os;
  }
};

/// Stream out the image statistics
inline std::ostream& operator<<(std::ostream& os, const image_stats& stats) {
  return stats.Print(os);
}

/// Compute the pixel statistics for a mono channel image
bool get_single_channel_stats(const cv::Mat& img, image_stats& stats);

/// Compute the pixel statistics of an image per channel
bool get_channel_stats(const cv::Mat& img, std::vector<image_stats>& stats);


} // end of namespace patchworks
} // end of namespace lbann
#endif // _PATCHWORKS_STATS_INCLUDED_
#endif // LBANN_HAS_OPENCV
