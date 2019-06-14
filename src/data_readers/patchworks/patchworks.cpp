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
// patchworks.cpp - LBANN PATCHWORKS main interface implementation
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS main interface implementation
 *  - includes the main interface function definitions
 */

#include "lbann/data_readers/patchworks/patchworks.hpp"

#ifdef LBANN_HAS_OPENCV
#include "lbann/utils/random.hpp"
#include "lbann/data_readers/patchworks/patchworks_stats.hpp"

namespace lbann {
namespace patchworks {

#if _PATCHWORKS_STAT_FLOAT_ == 32
#define _f f
#elif _PATCHWORKS_STAT_FLOAT_ == 64
#define _f
#else
#error need to set _PATCHWORKS_STAT_FLOAT_
#endif

std::pair<double,double> check_min_max(const cv::Mat& _img) {
  cv::Mat img = _img.clone();

  double maxVal = 0.0;
  double minVal = 0.0;
  const int nCh = img.channels();

  img.reshape(1);
  cv::minMaxLoc(img, &minVal, &maxVal, nullptr, nullptr);
  img.reshape(nCh);

  //std::cout << "min max : " << minVal << ' ' << maxVal << std::endl;
  return std::make_pair(minVal, maxVal);
}

cv::Mat correct_chromatic_aberration(const cv::Mat& _img) {
  if (_img.channels() != 3) {
    return _img.clone();
  }

  const int img_depth = _img.depth();

  std::pair<double, double> range_org = check_min_max(_img);

  cv::Mat img; // float matrix

  _img.convertTo(img, _PW_CV_FP_);

  static const pw_fp_t a[3] = {-1.0 _f, 2.0 _f, -1.0 _f}; // BGR order
  static const pw_fp_t aa = a[0]*a[0] + a[1]*a[1]+ a[2]*a[2];
  // A = a'*a/(a*a')
  //static const pw_fp_t A[3][3] = {{a[0]*a[0]/aa, a[0]*a[1]/aa,  a[0]*a[2]/aa},
  //                                {a[1]*a[0]/aa, a[1]*a[1]/aa,  a[1]*a[2]/aa},
  //                                {a[2]*a[0]/aa, a[2]*a[1]/aa,  a[2]*a[2]/aa}};
  // B = (I - A)'
  static const pw_fp_t B[3][3] = {{1.0 _f-a[0] *a[0]/aa, a[0] *a[1]/aa,  a[0] *a[2]/aa},
    {a[1] *a[0]/aa, 1.0 _f-a[1] *a[1]/aa,  a[1] *a[2]/aa},
    {a[2] *a[0]/aa, a[2] *a[1]/aa,  1.0 _f-a[2] *a[2]/aa}
  };

  cv::MatIterator_<pw_cv_vec3> it = img.begin<pw_cv_vec3>();
  cv::MatIterator_<pw_cv_vec3> itend = img.end<pw_cv_vec3>();

  for ( ; it != itend; ++it) {
    const auto b0 = static_cast<pw_fp_t>((*it)[0]);
    const auto g0 = static_cast<pw_fp_t>((*it)[1]);
    const auto r0 = static_cast<pw_fp_t>((*it)[2]);

    pw_fp_t b = b0 * B[0][0] + g0 * B[1][0] + r0 * B[2][0];
    pw_fp_t g = b0 * B[0][1] + g0 * B[1][1] + r0 * B[2][1];
    pw_fp_t r = b0 * B[0][2] + g0 * B[1][2] + r0 * B[2][2];

    //std::cout << r0 << ' ' << g0 << ' ' << b0 << " " << r << ' ' << g << ' ' << b << std::endl;
    (*it) = pw_cv_vec3(b,g,r);
  }

  std::pair<double, double> range_new = check_min_max(img);
  cv::Mat img_final;
  //(x-range_new.first)*(range_org.second-range_org.first)/(range_new.second-range_new.first) + range_org.first;
  const double alpha = (range_org.second-range_org.first)/(range_new.second-range_new.first);
  const double beta = range_org.first - range_new.first * alpha;
  img.convertTo(img_final, img_depth, alpha, beta);

  //std::pair<double, double> range_final = check_min_max(img_final);

  return img_final;
}

cv::Mat drop_2channels(const cv::Mat& _img) {
  if (_img.channels() != 3) {
    return _img.clone();
  }

  const int img_depth = _img.depth();

  cv::Mat img; // pw_fp_t matrix
  _img.convertTo(img, _PW_CV_FP_);

  // compute channel to remain
  pw_fp_t m[3] = {0.0 _f, 0.0 _f, 0.0 _f};

  ::lbann::rng_gen& gen = ::lbann::get_io_generator();

  std::uniform_int_distribution<int> rg_ch(0, 2);
  const int chosenCh = rg_ch(gen);

  m[chosenCh] = 1.0 _f;

  // compute white noise
  std::vector<image_stats> stats;
  get_channel_stats(_img, stats);

  const auto avg = static_cast<pw_fp_t>(stats[chosenCh].avg);
  const auto dev = static_cast<pw_fp_t>(stats[chosenCh].stdev/100.0);
  pw_fp_t avgs[3] = {avg, avg, avg};
  pw_fp_t devs[3] = {dev, dev, dev};

  std::normal_distribution<pw_fp_t> rg_ch0(avgs[0], devs[0]);
  std::normal_distribution<pw_fp_t> rg_ch1(avgs[1], devs[1]);
  std::normal_distribution<pw_fp_t> rg_ch2(avgs[2], devs[2]);

  cv::MatIterator_<pw_cv_vec3> it = img.begin<pw_cv_vec3>();
  cv::MatIterator_<pw_cv_vec3> itend = img.end<pw_cv_vec3>();

  for ( ; it != itend; ++it) {
    const auto b0 = static_cast<pw_fp_t>((*it)[0]);
    const auto g0 = static_cast<pw_fp_t>((*it)[1]);
    const auto r0 = static_cast<pw_fp_t>((*it)[2]);

#if 1
    pw_fp_t b = b0*m[0] + (1.0-m[0])*rg_ch0(gen);
    pw_fp_t g = g0*m[1] + (1.0-m[1])*rg_ch1(gen);
    pw_fp_t r = r0*m[2] + (1.0-m[2])*rg_ch2(gen);
#else
    pw_fp_t b = b0*m[0];
    pw_fp_t g = g0*m[1];
    pw_fp_t r = r0*m[2];
#endif

    //std::cout << r0 << ' ' << g0 << ' ' << b0 << " " << r << ' ' << g << ' ' << b << std::endl;
    (*it) = pw_cv_vec3(b,g,r);
  }

  cv::Mat img_final;
  img.convertTo(img_final, img_depth);

  return img_final;
}

} // end of namespace patchworks
} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
