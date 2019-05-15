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
// patchworks_stats.cpp - LBANN PATCHWORKS implementation for pixel statistics
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS implementation for pixel statistics
 */

#include "lbann/data_readers/patchworks/patchworks_stats.hpp"
#ifdef LBANN_HAS_OPENCV

namespace lbann {
namespace patchworks {

bool get_single_channel_stats(const cv::Mat& _img, image_stats& stats) {
  if (_img.channels() != 1) {
    return false;
  }

  cv::Mat img; // pw_fp_t matrix
  _img.convertTo(img, _PW_CV_FP_);

  cv::MatIterator_<pw_fp_t> itBegin = img.begin<pw_fp_t>();
  cv::MatIterator_<pw_fp_t> itEnd = img.end<pw_fp_t>();

  const auto typeZero = static_cast<pw_fp_t>(0);

  double sum = 0.0;

  std::vector<pw_fp_t> data(itBegin, itEnd);
  stats.cnt = data.size();
  if (stats.cnt == 0u) {
    return false;
  }

  std::sort(data.begin(), data.end());
  if (data[0] < typeZero) {
    return false;
  }

  stats.max = data.back();
  stats.min = data[0];

  std::vector<pw_fp_t>::const_iterator itbeg = data.begin();
  std::vector<pw_fp_t>::const_iterator itend = data.end();
  std::vector<pw_fp_t>::const_iterator itbegNZ = std::upper_bound(data.begin(), data.end(), static_cast<pw_fp_t>(0));

  stats.cntZeros = std::distance(itbeg, itbegNZ);
  stats.minNZ = *itbegNZ;

  const size_t nnz = stats.cnt - stats.cntZeros;
  const size_t halfPointNZ = nnz/2;
  const size_t halfPoint   = stats.cnt/2;
  auto itMedNZ = itbegNZ;
  auto itMed   = itbeg;
  std::advance(itMedNZ, halfPointNZ);
  std::advance(itMed,   halfPoint);

  stats.medianNZ = *itMedNZ;
  stats.median   = *itMed;

  auto it = itbegNZ;
  for( ; it != itend; ++it) {
    sum += *it;
  }

  stats.avg = sum/stats.cnt;
  if (nnz == 0u) {
    stats.avgNZ = stats.avg;
  } else {
    stats.avgNZ = sum/nnz;
  }

  double var = 0.0;
  double varNZ = 0.0;
  it = itbegNZ;

  for(it = itbeg; it != itbegNZ; ++it) {
    const double dev = (*it-stats.avg);
    var += dev*dev;
  }

  for( ; it != itend; ++it) {
    const double dev = (*it-stats.avg);
    var += dev*dev;
    const double devNZ = (*it-stats.avgNZ);
    varNZ += devNZ*devNZ;
  }

  stats.stdev = sqrt(var/stats.cnt);
  stats.stdevNZ = sqrt(varNZ/nnz);

  return true;
}

bool get_channel_stats(const cv::Mat& img, std::vector<image_stats>& stats) {
  if (img.data == nullptr) {
    std::cout << "get_channel_stats(): img not set" << std::endl;
    return false;
  }

  const int nCh = img.channels();
  std::vector<cv::Mat> imgCh; // image data per channel
  cv::split(img, imgCh); // split the image into individual channels

  stats.clear();
  stats.resize(nCh);

  bool ok = true;

  for (int ch=0; ok && (ch < img.channels()); ++ch) { // compute statistics per channel
    ok = get_single_channel_stats(imgCh[ch], stats[ch]);
  }

  if (!ok) {
    std::cout << "Failed to get stats" << std::endl;
  }
  return ok;
}

} // end of namespace patchworks
} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
