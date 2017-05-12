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
// lbann_cv_preprocessor .cpp .hpp - Prerpocessing functions for images
//                                   in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_PREPROCESSOR_HPP
#define LBANN_CV_PREPROCESSOR_HPP

#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#include "lbann/lbann_base.hpp" // DataType
#include "lbann/utils/lbann_mild_exception.hpp"

#ifdef __LIB_OPENCV
namespace lbann
{

class cv_preprocessor
{
 public:
  /**
   * Define the type of normalization methods available. 
   * z-score method is essentially the combination of mean subtraction and unit variance
   */
  enum normalization_type {_none=0, _u_scale=1, _mean_sub=2, _unit_var=4, _z_score=6};

 protected:
  // Parameters for normalization

  /// Whether to normalize to 0 mean.
  bool m_mean_subtraction;
  /// Whether to normalize to unit variance.
  bool m_unit_variance;
  /// Whether to scale to [0, 1].
  bool m_unit_scale;
  /// Whether to normalize via z-score.
  bool m_z_score;

  /// Set a normalization bit flag
  virtual normalization_type set_normalization_bits(const normalization_type ntype, const normalization_type flag) const
  { return static_cast<normalization_type>(static_cast<uint32_t>(ntype) | static_cast<uint32_t>(flag)); }

  /// Mask normalization bits
  virtual normalization_type mask_normalization_bits(const normalization_type ntype, const normalization_type flag) const
  { return static_cast<normalization_type>(static_cast<uint32_t>(ntype) & static_cast<uint32_t>(flag)); }

 public:

  cv_preprocessor(void)
  : m_mean_subtraction(false), m_unit_variance(false), m_unit_scale(false), m_z_score(false) {}

  /// Enable a particular normalization method
  virtual normalization_type& set_normalization_type(normalization_type& ntype, const normalization_type flag) const;

  /// Whether to subtract the per-channel and per-sample mean.
  void subtract_mean(bool b) { m_mean_subtraction = b; }
  /// Whether to normalize to unit variance, per-channel and per-sample.
  void unit_variance(bool b) { m_unit_variance = b; }
  /// Whether to scale to [0, 1]
  void unit_scale(bool b) { m_unit_scale = b; }
  /// Whether to normalize by z-scores, per-channel and per-sample.
  void z_score(bool b) { m_z_score = b; }

  // TODO: work-in-progress
  bool augment(cv::Mat& image) { return true; }

  /// Combine the normalizations emabled and define a linear transform to address them all
  virtual bool determine_normalization(const cv::Mat& image,
          std::vector<double>& alpha, std::vector<double>& beta) const;

  virtual bool normalize(cv::Mat& image,
          const std::vector<double>& alpha, const std::vector<double>& beta) const;

  virtual bool unnormalize(cv::Mat& image,
          const std::vector<double>& alpha, const std::vector<double>& beta) const;


  template<class InputIterator, class OutputIterator>
  static OutputIterator scale(InputIterator first, InputIterator last, OutputIterator result,
                        const std::vector<double>& alpha, const std::vector<double>& beta);

  template<typename Tsrc, typename Tdst>
  static bool scale_with_known_type (cv::Mat& image,
              const std::vector<double>& alpha, const std::vector<double>& beta);
  /**
   * Scale an image using a scaling parameter alpha and a shift parameter beta.
   * The resultant image will contains channel values of LBANN's DataType.
   */
  static bool scale(cv::Mat& image,
       const std::vector<double>& alpha, const std::vector<double>& beta);


  template<typename T>
  static bool compute_mean_stddev_with_known_type(const cv::Mat& image,
       std::vector<double>& mean, std::vector<double>& stddev, cv::Mat mask);

  /// Compute the per-channel and per-sample mean and standard deviation
  static bool compute_mean_stddev(const cv::Mat& image,
       std::vector<double>& mean, std::vector<double>& stddev, cv::Mat mask=cv::Mat());
};


inline bool cv_preprocessor::normalize(cv::Mat& image,
  const std::vector<double>& alpha, const std::vector<double>& beta) const
{
  return scale(image, alpha, beta);
}


/**
 * Transform linearly while copying data from one sequential container to another
 * The transformation is alpha[ch]*input[ch] + beta[ch] -> output[ch]
 * @param first  The beginning of the input interator
 * @param last   The last of the input iterator
 * @param result The beginning of the output iterator
 * @param alpha  per-channel scaling parameter for linear transform
 * @param beta   per-channel shifting parameter for linear transform
 * @return the last of output iterator
 */
template<class InputIterator, class OutputIterator>
inline OutputIterator cv_preprocessor::scale(
  InputIterator first, InputIterator last, OutputIterator result,
  const std::vector<double>& alpha, const std::vector<double>& beta)
{
  _LBANN_MILD_EXCEPTION(alpha.size() != beta.size(), \
                        "Inconsistent scaling parameters.", result)

  const size_t NCh = alpha.size();
  bool trivial_alpha = true;
  bool trivial_beta = true;

  for (size_t ch=0u; ch < NCh; ++ch) {
    trivial_alpha = trivial_alpha && (alpha[ch] == 1.0);
    trivial_beta  = trivial_beta  && (beta[ch] == 0.0);
  }

  if (trivial_alpha && trivial_beta) {
    if ((typeid(*first) == typeid(*result)) &&
        (reinterpret_cast<const void*>(&(*first)) ==
         reinterpret_cast<const void*>(&(*result))))
        // This way, it works both for iterator and for pointer
    {
      std::advance(result, std::distance(first,last));
      return result;
    } else {
      return std::copy(first, last, result);
    }
  }

  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  // At this point NCh should not be zero because alpha.size() == beta.size()
  // and both are not trivial.
  if (NCh == 1) {
    const double a = alpha[0];
    const double b = beta[0];

    while (first != last) {
      *result = cv::saturate_cast<T>(a * (*first) + b);
      ++result; ++first;
    }
  } else {
    size_t ch = 0u;

    while (first != last) {
      *result = cv::saturate_cast<T>(alpha[ch] * (*first) + beta[ch]);
      ++result; ++first; ++ch;
      ch = (ch % NCh);
    }
  }
  return result;
}


/**
 * Linear transform image pixels by scaling parameters given for each channel
 * The transformation is alpha[ch]*input[ch] + beta[ch] -> output[ch].
 * The first template parameter is the channel value type of the input image.
 * The second one is the channel value type desired for the output image.
 *
 * @param image  The image to be modified, which is the input and also the ouput.
 * @param alpha  per-channel scaling parameter for linear transform
 * @param beta   per-channel shifting parameter for linear transform
 * @return true if successful. The input image will be modified to a new one.
 */
template<typename Tsrc, typename Tdst>
inline bool cv_preprocessor::scale_with_known_type(cv::Mat& image,
  const std::vector<double>& alpha, const std::vector<double>& beta)
{
  const unsigned int Width  = static_cast<unsigned int>(image.cols);
  const unsigned int Height = static_cast<unsigned int>(image.rows);
  const unsigned int NCh    = static_cast<unsigned int>(image.channels());
  if (((alpha.size() > 0u) && (alpha.size() != NCh)) ||
      ((beta.size() > 0u) && (beta.size() != NCh)))
    return false;

  
  // overwrite the storage of the source image if the source and the result have
  // the same data type. Otherwise, create a new image for the result and replace
  // the input with the input at the end.
  if (std::is_same<Tsrc, Tdst>::value) {
    if (image.isContinuous()) {
      scale(reinterpret_cast<const Tsrc*>(image.datastart),
            reinterpret_cast<const Tsrc*>(image.dataend),
            reinterpret_cast<Tsrc*>(image.data), alpha, beta);
    } else {
      const unsigned int stride = Height*NCh;
      for (unsigned int i = 0u; i < Height; ++i) {
        Tsrc* optr = reinterpret_cast<Tsrc*>(image.ptr<Tsrc>(i));
        const Tsrc* iptr = optr;
        scale(iptr, iptr+stride, optr, alpha, beta);
      }
    }
  } else {
    cv::Mat image_out = cv::Mat(Height, Width, CV_MAKETYPE(cv::DataType<Tdst>::depth, NCh));

    if (image.isContinuous()) {
      scale(reinterpret_cast<const Tsrc*>(image.datastart),
            reinterpret_cast<const Tsrc*>(image.dataend),
            reinterpret_cast<Tdst*>(image_out.data), alpha, beta);
    } else {
      const unsigned int stride = Height*NCh;
      Tdst* ptr_out = reinterpret_cast<Tdst*>(image_out.data);
      for (unsigned int i = 0u; i < Height; ++i, ptr_out += stride) {
        const Tsrc* ptr = reinterpret_cast<Tsrc*>(image.ptr<Tsrc>(i));
        scale(ptr, ptr+stride, ptr_out, alpha, beta);
      }
    }
    image = image_out;
  }
  return true;
}


/**
 * Compute the per-channel and per-sample mean and standard deviation
 * for a sample image of channel value type T
 */
template<typename T>
inline bool cv_preprocessor::compute_mean_stddev_with_known_type(const cv::Mat& image,
  std::vector<double>& mean, std::vector<double>& stddev, cv::Mat mask)
{
  mean.clear();
  stddev.clear();
  if (image.empty()) return false;

  const int NCh = image.channels();
  const int num_pixels = image.rows * image.cols;
  double sum[NCh] = {0.0, };
  double sqsum[NCh] = {0.0, };
  double shift[NCh] = {0.0,};

  for (int ch = 0; ch < NCh; ++ch) {
    sum[ch] = 0.0;
    sqsum[ch] = 0.0;
    const T* ptr = reinterpret_cast<const T*>(image.datastart);
    shift[ch] = static_cast<double>(*(ptr+ch));
  }

  mean.resize(NCh);
  stddev.resize(NCh);

  if (image.isContinuous()) {
    const T* ptr = reinterpret_cast<const T*>(image.datastart);
    const T* const ptrend = reinterpret_cast<const T* const>(image.dataend);

    int ch = 0;
    do {
      const double diff = (*ptr - shift[ch]);
      sum[ch] += diff;
      sqsum[ch] += diff*diff;
      ++ch;
      ch = ch % NCh;
    } while ((++ptr) != ptrend);

    for (int ch = 0; ch < NCh; ++ch) {
      const double shifted_mean = sum[ch] / num_pixels;
      mean[ch] = shifted_mean + shift[ch];
      stddev[ch] = sqrt(sqsum[ch]/num_pixels - shifted_mean * shifted_mean);
    }
  } else {
    const int stride = image.cols*NCh;
    const int Height = image.rows;

    for (int i = 0; i < Height; ++i) {
      const T* ptr = reinterpret_cast<const T*>(image.ptr<const T>(i));
      const T* const ptrend = ptr + stride;

      int ch = 0;
      do {
        const double diff = (*ptr - shift[ch]);
        sum[ch] += diff;
        sqsum[ch] += diff*diff;
        ++ch;
        ch = ch % NCh;
      } while ((++ptr) != ptrend);
    }
  
    for (int ch = 0; ch < NCh; ++ch) {
      const double shifted_mean = sum[ch] / num_pixels;
      mean[ch] = shifted_mean + shift[ch];
      stddev[ch] = sqrt(sqsum[ch]/num_pixels - shifted_mean*shifted_mean);
    }
  }
 #if 0
  for (size_t ch = 0u; ch < NCh; ++ch)
    std::cout << "channel " << ch << "\tmean " << mean[ch] << "\tstddev " << stddev[ch] << std::endl;
 #endif
  return true;
}

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_PREPROCESSOR_HPP
