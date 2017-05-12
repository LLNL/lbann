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
// lbann_cv_preprocessor .cpp .hpp - Image I/O utility functions
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_cv_preprocessor.hpp"
#include <cmath> //fabs

#ifdef __LIB_OPENCV
namespace lbann
{

cv_preprocessor::normalization_type& cv_preprocessor::set_normalization_type(
  normalization_type& ntype, const normalization_type flag) const
{
  return (ntype = set_normalization_bits(ntype, flag));
}


bool cv_preprocessor::determine_normalization(const cv::Mat& image,
  std::vector<double>& alpha, std::vector<double>& beta) const
{
  if (image.empty()) return false;

  normalization_type ntype = _none;
  if (m_scale)            set_normalization_type(ntype, _u_scale);
  if (m_mean_subtraction) set_normalization_type(ntype, _mean_sub);
  if (m_unit_variance)    set_normalization_type(ntype, _unit_var);
  if (m_z_score)          set_normalization_type(ntype, _z_score);

  double unit_scale = 1.0;
  double largest = 1.0;

  //std::cout << "normalization setup code: " << static_cast<uint32_t>(ntype) << std::endl;

  if (!m_z_score && m_scale) {
    switch(image.depth()) {
      case CV_8U:  largest = std::numeric_limits<uint8_t>::max();  break;
      case CV_8S:  largest = std::numeric_limits<int8_t>::max();   break;
      case CV_16U: largest = std::numeric_limits<uint16_t>::max(); break;
      case CV_16S: largest = std::numeric_limits<int16_t>::max();  break;
      case CV_32S: largest = std::numeric_limits<int32_t>::max();  break;
      default: return false;
      // Currently, do nothing for non-integral types. However, a set of scaling
      // paramters can be added to the argument list of this function.
    }
    unit_scale = 1.0/largest;
  }

  std::vector<double> mean;
  std::vector<double> stddev;
  const normalization_type code_wo_uscale = mask_normalization_bits(ntype, _z_score);
  const size_t NCh = static_cast<size_t>(image.channels());

  if (code_wo_uscale != _none) {
    if (!compute_mean_stddev(image, mean, stddev) || (NCh != mean.size()))
      return false;
  }

  alpha.resize(NCh);
  beta.resize(NCh);

  switch (code_wo_uscale) {
    case _none: // Note that mean.size() is zero in this case
        for (size_t ch=0u; ch < NCh; ++ch) {
          alpha[ch] = unit_scale;
          beta[ch]  = 0.0;
        }
        break;
    case _mean_sub:
        for (size_t ch=0u; ch < NCh; ++ch) {
          alpha[ch] = unit_scale;
          beta[ch]  = - unit_scale * mean[ch];
        }
        break;
    case _unit_var:
        for (size_t ch=0u; ch < NCh; ++ch) {
          if (stddev[ch] > fabs(mean[ch])*(1e-7)) {
            alpha[ch] = 1.0/stddev[ch];
            beta[ch]  = unit_scale * mean[ch] - mean[ch]/stddev[ch];
          } else {
            alpha[ch] = unit_scale;
            beta[ch]  = 0.0;
          }
        }
        break;
    case _z_score:
        for (size_t ch=0u; ch < NCh; ++ch) {
          if (stddev[ch] > fabs(mean[ch])*(1e-7)) {
            alpha[ch] = 1.0/stddev[ch];
            beta[ch]  = - mean[ch]/stddev[ch];
          } else {
            alpha[ch] = 0.0;
            beta[ch]  = 0.0;
          }
        }
        break;
    default: return false;
  }
  return true;
}


bool cv_preprocessor::scale(cv::Mat& image,
  const std::vector<double>& alpha, const std::vector<double>& beta)
{
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  switch(image.depth()) {
    case CV_8U:  return scale_with_known_type<_depth_type(CV_8U),  ::DataType>(image, alpha, beta);
    case CV_8S:  return scale_with_known_type<_depth_type(CV_8S),  ::DataType>(image, alpha, beta);
    case CV_16U: return scale_with_known_type<_depth_type(CV_16U), ::DataType>(image, alpha, beta);
    case CV_16S: return scale_with_known_type<_depth_type(CV_16S), ::DataType>(image, alpha, beta);
    case CV_32S: return scale_with_known_type<_depth_type(CV_32S), ::DataType>(image, alpha, beta);
    case CV_32F: return scale_with_known_type<_depth_type(CV_32F), ::DataType>(image, alpha, beta);
    case CV_64F: return scale_with_known_type<_depth_type(CV_64F), ::DataType>(image, alpha, beta);
  }
  return false;
}


bool cv_preprocessor::compute_mean_stddev(const cv::Mat& image,
  std::vector<double>& mean, std::vector<double>& stddev, cv::Mat mask)
{
  bool ok = true;
 #if 0
  cv::meanStdDev(image, mean, stddev, mask);
  mean.resize(image.channels());
  stddev.resize(image.channels());
 #else
  if (image.empty()) return false;
  switch(image.depth()) {
    case CV_8U:  ok = compute_mean_stddev_with_known_type<_depth_type(CV_8U) >(image, mean, stddev, mask); break;
    case CV_8S:  ok = compute_mean_stddev_with_known_type<_depth_type(CV_8S) >(image, mean, stddev, mask); break;
    case CV_16U: ok = compute_mean_stddev_with_known_type<_depth_type(CV_16U)>(image, mean, stddev, mask); break;
    case CV_16S: ok = compute_mean_stddev_with_known_type<_depth_type(CV_16S)>(image, mean, stddev, mask); break;
    case CV_32S: ok = compute_mean_stddev_with_known_type<_depth_type(CV_32S)>(image, mean, stddev, mask); break;
    case CV_32F: ok = compute_mean_stddev_with_known_type<_depth_type(CV_32F)>(image, mean, stddev, mask); break;
    case CV_64F: ok = compute_mean_stddev_with_known_type<_depth_type(CV_64F)>(image, mean, stddev, mask); break;
  }
 #endif
  //for (size_t ch = 0u; ch < mean.size(); ++ch)
  //  std::cout << "channel " << ch << "\tmean " << mean[ch] << "\tstddev " << stddev[ch] << std::endl;
  return (ok && (mean.size() == stddev.size()));
}

} // end of namespace lbann
#endif // __LIB_OPENCV
