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
// lbann_cv_normalizer .cpp .hpp - Normalizing functions for images
//                                 in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_NORMALIZER_HPP
#define LBANN_CV_NORMALIZER_HPP

#include "cv_transform.hpp"
#include "patchworks/patchworks_opencv.hpp"
#include "lbann/base.hpp" // DataType
#include "lbann/utils/mild_exception.hpp"

#ifdef __LIB_OPENCV
namespace lbann {
/**
 *  Modifies the channel values of each pixel according to the chosen normalization
 *  strategies:
 *  - Standardize to 0 mean
 *  - Standardize to unit variance
 *  - Scale to the range [0, 1]
 *  - Normalize via z-score
 *
 *  Combine these strategies into a single per-pixel linear transform, and
 *  process them all at once.
 *  It tries to replace the values in place if possible, rather
 *  than creating a new copy of data, especially, if the channel data type of
 *  source image is the same as that of the resultant image.
 */
class cv_normalizer : public cv_transform {
 public:
  /** This is the interim type of input values computed from image data
   *  It does not have to be the same as the type of the values stored, i.e., DataType.
   */
  typedef DataType ComputeType;
  //typedef double ComputeType;
  /**
   * Define the type of normalization methods available.
   * z-score method is essentially the combination of mean subtraction and unit variance
   */
  enum normalization_type {_none=0, _u_scale=1, _mean_sub=2, _unit_var=4, _z_score=6};
  typedef std::pair<ComputeType, ComputeType> channel_trans_t;

 protected:
  // --- Parameters for normalization ---
  /// Whether to normalize to 0 mean.
  bool m_mean_subtraction;
  /// Whether to normalize to unit variance.
  bool m_unit_variance;
  /// Whether to scale to [0, 1].
  bool m_unit_scale;
  /// Whether to normalize via z-score.
  bool m_z_score;


  // --- normalizing transform determined ---
  /**
   *  The parameter to use for linearly transforming channel values of each pixel as:
   *  new_value[ch] = cv::saturate_cast<T>(m_trans[ch].first*value[ch] + m_trans[ch].second)
   */
  std::vector<channel_trans_t> m_trans;


  /// Set a normalization bit flag
  virtual normalization_type set_normalization_bits(const normalization_type ntype, const normalization_type flag) const {
    return static_cast<normalization_type>(static_cast<uint32_t>(ntype) | static_cast<uint32_t>(flag));
  }

  /// Mask normalization bits
  virtual normalization_type mask_normalization_bits(const normalization_type ntype, const normalization_type flag) const {
    return static_cast<normalization_type>(static_cast<uint32_t>(ntype) & static_cast<uint32_t>(flag));
  }

  /// Enable a particular normalization method
  virtual normalization_type& set_normalization_type(normalization_type& ntype, const normalization_type flag) const;

  /// Check if there is a reason to enable. (i.e., any option set)
  virtual bool check_to_enable() const;

 public:

  cv_normalizer();
  cv_normalizer(const cv_normalizer& rhs);
  cv_normalizer& operator=(const cv_normalizer& rhs);
  virtual cv_normalizer *clone() const;

  virtual ~cv_normalizer() {}

  /// Set the parameters all at once
  virtual void set(const bool meansub, const bool unitvar, const bool unitscale, const bool zscore);

  /// Whether to subtract the per-channel and per-sample mean.
  void subtract_mean(bool b) {
    m_mean_subtraction = b;
  }
  /// Whether to normalize to unit variance, per-channel and per-sample.
  void unit_variance(bool b) {
    m_unit_variance = b;
  }
  /// Whether to scale to [0, 1]
  void unit_scale(bool b) {
    m_unit_scale = b;
  }
  /// Whether to normalize by z-scores, per-channel and per-sample.
  void z_score(bool b) {
    m_z_score = b;
  }

  /// Reset all the paramters to the default values
  virtual void reset();

  /// Returns the channel-wise scaling parameter for normalization transform
  std::vector<channel_trans_t> transform() const {
    return (m_enabled? m_trans : std::vector<channel_trans_t>());
  }

  /**
   * Combine the normalizations enabled and define a linear transform
   * per pixel to address them all. If successful, the tranform is enabled.
   * If not, it is disabled.
   * @return false if not enabled or unsuccessful.
   */
  virtual bool determine_transform(const cv::Mat& image);

  /**
   * Apply the normalization defined as a linear tranform per pixel.
   * As this method is executed, the transform becomes deactivated.
   * @return false if not successful.
   */
  virtual bool apply(cv::Mat& image);

  /// Set a pre-determined normalization transform.
  void set_transform(const std::vector<channel_trans_t>& t);
  /**
   * Reverse the normalization done as x' = alpha*x + beta by
   * x = (x'- beta)/alpha
   * If successful, the tranform is enabled. If not, it is disabled.
   * @return false if not enabled or unsuccessful.
   */
  bool determine_inverse_transform();

  // utilities
  template<class InputIterator, class OutputIterator>
  static OutputIterator scale(InputIterator first, InputIterator last, OutputIterator result,
                              const std::vector<channel_trans_t> trans);

  template<typename Tsrc, typename Tdst>
  static bool scale_with_known_type(cv::Mat& image, const std::vector<channel_trans_t>& trans);

  /**
   * Scale an image using a set of parameters for linearly transforming channel
   * values per pixel.
   * The resultant image will contain channel values of LBANN's DataType.
   */
  static bool scale(cv::Mat& image, const std::vector<channel_trans_t>& trans);


  template<typename T>
  static bool compute_mean_stddev_with_known_type(const cv::Mat& image,
      std::vector<ComputeType>& mean, std::vector<ComputeType>& stddev, cv::InputArray mask);

  /// Compute the per-channel and per-sample mean and standard deviation
  static bool compute_mean_stddev(const cv::Mat& image,
                                  std::vector<ComputeType>& mean, std::vector<ComputeType>& stddev,
                                  cv::InputArray mask=cv::noArray());

  virtual std::ostream& print(std::ostream& os) const;
};


/**
 * Linearly transform each value while copying it from one sequential container
 * to another, which may be the same container if the type of the initial value
 * and that of the result are the same.
 * The transformation is alpha[ch]*input[ch] + beta[ch] -> output[ch]
 * @param first  The beginning of the input interator
 * @param last   The last of the input iterator
 * @param result The beginning of the output iterator
 * @param trans  Parameters for linearly transforming channel values per pixel
 * @return the last of output iterator
 */
template<class InputIterator, class OutputIterator>
inline OutputIterator cv_normalizer::scale(
  InputIterator first, InputIterator last, OutputIterator result,
  const std::vector<channel_trans_t> trans) {
  const size_t NCh = trans.size();
  bool trivial_alpha = true;
  bool trivial_beta = true;

  for (size_t ch=0u; ch < NCh; ++ch) {
    trivial_alpha = trivial_alpha && (trans[ch].first  == 1.0);
    trivial_beta  = trivial_beta  && (trans[ch].second == 0.0);
  }

  if (trivial_alpha && trivial_beta) {
    if ((typeid(*first) == typeid(*result)) &&
        (reinterpret_cast<const void *>(&(*first)) ==
         reinterpret_cast<const void *>(&(*result))))
      // This way, it works both for iterator and for pointer
    {
      std::advance(result, std::distance(first,last));
      return result;
    } else {
      return std::copy(first, last, result);
    }
  }

  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  // At this point NCh should not be zero because both alpha and beta are not trivial.
  if (NCh == 1) {
    const ComputeType a = trans[0].first;
    const ComputeType b = trans[0].second;

    while (first != last) {
      *result = cv::saturate_cast<T>(a * (*first) + b);
      ++result;
      ++first;
    }
  } else {
    size_t ch = 0u;

    while (first != last) {
      *result = cv::saturate_cast<T>(trans[ch].first * (*first) + trans[ch].second);
      ++result;
      ++first;
      ++ch;
      ch = (ch % NCh);
    }
  }
  return result;
}


/**
 * Linear transform image pixels by scaling parameters given for each channel
 * The transformation is trans[ch].first*input[ch] + trans[ch].second -> output[ch].
 * The first template parameter is the channel value type of the input image.
 * The second one is the channel value type desired for the output image.
 *
 * @param image  The image to be modified, which is the input and also the ouput.
 * @param trans  Parameters for linearly transforming channel values per pixel
 * @return true if successful. The input image will be modified to a new one.
 */
template<typename Tsrc, typename Tdst>
inline bool cv_normalizer::scale_with_known_type(cv::Mat& image,
    const std::vector<channel_trans_t>& trans) {
  const unsigned int Width  = static_cast<unsigned int>(image.cols);
  const unsigned int Height = static_cast<unsigned int>(image.rows);
  const unsigned int NCh    = static_cast<unsigned int>(image.channels());
  if ((trans.size() > 0u) && (trans.size() != NCh)) {
    return false;
  }


  // overwrite the storage of the source image if the source and the result have
  // the same data type. Otherwise, create a new image for the result and replace
  // the input with the input at the end.
  if (std::is_same<Tsrc, Tdst>::value) {
    if (image.isContinuous()) {
      scale(reinterpret_cast<const Tsrc *>(image.datastart),
            reinterpret_cast<const Tsrc *>(image.dataend),
            reinterpret_cast<Tsrc *>(image.data), trans);
    } else {
      const unsigned int stride = Height*NCh;
      for (unsigned int i = 0u; i < Height; ++i) {
        Tsrc *optr = reinterpret_cast<Tsrc *>(image.ptr<Tsrc>(i));
        const Tsrc *iptr = optr;
        scale(iptr, iptr+stride, optr, trans);
      }
    }
  } else {
    cv::Mat image_out = cv::Mat(Height, Width, CV_MAKETYPE(cv::DataType<Tdst>::depth, NCh));

    if (image.isContinuous()) {
      scale(reinterpret_cast<const Tsrc *>(image.datastart),
            reinterpret_cast<const Tsrc *>(image.dataend),
            reinterpret_cast<Tdst *>(image_out.data), trans);
    } else {
      const unsigned int stride = Height*NCh;
      Tdst *ptr_out = reinterpret_cast<Tdst *>(image_out.data);
      for (unsigned int i = 0u; i < Height; ++i, ptr_out += stride) {
        const Tsrc *ptr = reinterpret_cast<Tsrc *>(image.ptr<Tsrc>(i));
        scale(ptr, ptr+stride, ptr_out, trans);
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
inline bool cv_normalizer::compute_mean_stddev_with_known_type(const cv::Mat& image,
    std::vector<ComputeType>& mean, std::vector<ComputeType>& stddev, cv::InputArray mask) {
  mean.clear();
  stddev.clear();
  if (image.empty()) {
    return false;
  }

  const int NCh = image.channels();
  const int num_pixels = image.rows * image.cols;
  ComputeType sum[NCh];
  ComputeType sqsum[NCh];
  ComputeType shift[NCh];

  for (int ch = 0; ch < NCh; ++ch) {
    sum[ch] = 0.0;
    sqsum[ch] = 0.0;
    const T *ptr = reinterpret_cast<const T *>(image.datastart);
    shift[ch] = static_cast<ComputeType>(*(ptr+ch));
  }

  mean.resize(NCh);
  stddev.resize(NCh);

  if (image.isContinuous()) {
    const T *ptr = reinterpret_cast<const T *>(image.datastart);
    const T *const ptrend = reinterpret_cast<const T *>(image.dataend);

    int ch = 0;
    do {
      const ComputeType diff = (*ptr - shift[ch]);
      sum[ch] += diff;
      sqsum[ch] += diff*diff;
      ++ch;
      ch = ch % NCh;
    } while ((++ptr) != ptrend);

    for (int c = 0; c < NCh; ++c) {
      const ComputeType shifted_mean = sum[c] / num_pixels;
      mean[c] = shifted_mean + shift[c];
      stddev[c] = sqrt(std::max(sqsum[c]/num_pixels - shifted_mean * shifted_mean, ComputeType(0)));
    }
  } else {
    const int stride = image.cols*NCh;
    const int Height = image.rows;

    for (int i = 0; i < Height; ++i) {
      const T *ptr = reinterpret_cast<const T *>(image.ptr<const T>(i));
      const T *const ptrend = ptr + stride;

      int ch = 0;
      do {
        const ComputeType diff = (*ptr - shift[ch]);
        sum[ch] += diff;
        sqsum[ch] += diff*diff;
        ++ch;
        ch = ch % NCh;
      } while ((++ptr) != ptrend);
    }

    for (int ch = 0; ch < NCh; ++ch) {
      const ComputeType shifted_mean = sum[ch] / num_pixels;
      mean[ch] = shifted_mean + shift[ch];
      stddev[ch] = sqrt(std::max(sqsum[ch]/num_pixels - shifted_mean*shifted_mean, ComputeType(0)));
    }
  }
  return true;
}

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_NORMALIZER_HPP
