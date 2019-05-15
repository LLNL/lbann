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
// cv_mean_extractor .cpp .hpp - accumulate mean over the image set
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_MEAN_EXTRACTOR_HPP
#define LBANN_CV_MEAN_EXTRACTOR_HPP

#include "cv_transform.hpp"
#include <type_traits>

#ifdef LBANN_HAS_OPENCV
namespace lbann {

/**
 *  Computes a cumulative pixel-wise average of a stream of images.
 *  It is assumed that the images have the same size and the same number of
 *  channels. However, they are not required to have the same channel depth.
 *  If a channel value is an integral type, it is normalized to a floating
 *  point number of type Float_T between 0 and 1 (inclusive at both ends).
 *  If a channel value is already in a floating point type, the value is used
 *  without normalization.
 *  Images accumulate per pixel and a mean image is obtained by dividing each
 *  pixel accumulation by the total number of images (if m_batch_size is larger
 *  than the number of all the images observed). The current mean of images can
 *  be obtained at any point during the operation by the member function
 *  extract<Channel_T>(). This returns the image normalized to the range of
 *  channel type, Channel_T. For example, if Channel_T is uint8_t, the range of
 *  mean values from 0.0 to 1.0 maps to the range from 0 to 256.
 *  To cope with a large number of images, one might rely on semi-moving average
 *  method. Up to m_batch_size number of images accumulate aa a batch while the
 *  moving average of batches is computed upon request by calling extract().
 *  This is particularly useful when Float_T is single precision with a limited
 *  number of bits to represent a wide range of numbers and the images have a
 *  large bit depth.
 */
class cv_mean_extractor : public cv_transform {
 public:
  /// type of image statistics value accumulated
  using Float_T = double;
  static const unsigned int m_default_batch_size = 65536u;

 protected:
  // --- configuration variables ---
  unsigned int m_batch_size; ///< number of samples per batch

  // --- state variables ---
  unsigned int m_batch_cnt; ///< number of complete batches
  unsigned int m_partial_cnt; ///< number of samples currently contributing towards a batch
  /// OpenCv type code used to create  m_sum and m_avg based on Float_T and the number of channels
  int m_type_code;
  cv::Mat m_sum; ///< partial batch accumulated so far
  cv::Mat m_avg; ///< cumulative moving average

  /// create the matrices for accumulating image statistics
  void create_matrices(const unsigned int width, const unsigned int height, const unsigned int n_ch);

 public:
  cv_mean_extractor();
  cv_mean_extractor(const cv_mean_extractor& rhs);
  cv_mean_extractor& operator=(const cv_mean_extractor& rhs);
  cv_mean_extractor *clone() const override;

  ~cv_mean_extractor() override {}

  void set(const unsigned int width, const unsigned int height, const unsigned int n_ch,
           const unsigned int batch_sz = cv_mean_extractor::m_default_batch_size);
  void set(const unsigned int batch_sz);
  void reset() override;

  bool determine_transform(const cv::Mat& image) override;
  /// The transform does not modify the image. Thus, this has no effect.
  bool determine_inverse_transform() override;
  bool apply(cv::Mat& image) override;

  template<typename Channel_T = uint8_t>
  cv::Mat extract() const;

  std::string get_type() const override { return "mean extractor"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;
};

/**
 * Convert the maxtrix representing the cumulative moving average of images
 * observed so far into an image with the channel type 'Channel_T'. The default
 * is uint8_t. If it is given as void, the matrix is returned as is.
 */
template<typename Channel_T>
inline cv::Mat cv_mean_extractor::extract() const {
  cv::Mat avg_so_far;
  if (m_partial_cnt == 0u) {
    avg_so_far = m_avg;
  } else {
    cv::addWeighted(m_avg, m_batch_cnt/static_cast<double>(m_batch_cnt+1),
                    m_sum, 1/static_cast<double>((m_batch_cnt + 1) * m_partial_cnt),
                    0.0, avg_so_far, m_type_code);
  }

  if (avg_so_far.empty()) return cv::Mat();

  if (std::is_void<Channel_T>::value) return avg_so_far;

  double minVal = 0.0;
  double maxVal = 0.0;
  cv::minMaxLoc(avg_so_far, &minVal, &maxVal, nullptr, nullptr);
  //const double max_channel_type = std::numeric_limits<Channel_T>::max();
  const double max_channel_type = depth_normalization<Channel_T>::inverse_factor();

  cv::Mat recovered;
  if ((minVal < 0.0) || (maxVal > 1.0)) {
    // This condition may rise either because of unnormalized images with raw
    // floating point values or because of precision error. In these cases,
    // the minimum value maps to 0 and the maximum value maps to the greatest
    // value of Channel_T
    const double range = maxVal-minVal;
    if (range == 0.0) return cv::Mat();
    const double alpha = max_channel_type/range;
    const double beta  = - alpha*minVal;
    avg_so_far.convertTo(recovered, cv_image_type<Channel_T>::T(),
                         alpha, beta);
  } else {
    // In this case, 0 maps to 0, and 1 maps to the greatest value of Channel_T
    avg_so_far.convertTo(recovered, cv_image_type<Channel_T>::T(),
                         max_channel_type, 0.0);
  }

  return recovered;
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_MEAN_EXTRACTOR_HPP
