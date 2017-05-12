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
// lbann_cv_process .cpp .hpp - structure that defines the operations
//                              on image data in opencv format
////////////////////////////////////////////////////////////////////////////////


#include "lbann/data_readers/lbann_cv_process.hpp"

#ifdef __LIB_OPENCV
namespace lbann
{

/**
 * Set the linear transform parameters for normalization, and let the actual
 * transform applied during copying from a cv::Mat  image to El::Matrix<DataType>
 * data to avoid reading the image twice.
 * @param _alpha The channel-wise scaling parameters for linear transform
 * @param _beta  The channel-wise shifting parameters for linear transform
 * @return Return false if the number of scaling and shifting parameters does not match.
 */
bool cv_process::set_to_normalize(const std::vector<double>& _alpha, const std::vector<double>& _beta)
{
  if (_alpha.size() != _beta.size()) return false;
  m_alpha = _alpha;
  m_beta = _beta;
  return true;
}

/**
 * In case that undoing normalization is required, this call lets it occur
 * during copying from El::Matrix<DataType> data to a cv::Mat image while
 * avoiding reading the image twice.
 */
bool cv_process::set_to_unnormalize(void)
{
  if (m_alpha.size() != m_beta.size()) return false;

  if ((m_alpha.size() == 0u) && (m_alpha_used.size() > 0u)) {
    m_alpha.swap(m_alpha_used);
    m_beta.swap(m_beta_used);
  }

  const size_t NCh = m_alpha.size();

  std::vector<double> alpha_reverse(NCh, 1.0);
  std::vector<double> beta_reverse(NCh, 0.0);

  for (size_t ch=0u; ch < NCh; ++ch) {
    if (m_alpha[ch] == 0.0) return false;
    alpha_reverse[ch] = 1.0/m_alpha[ch];
    beta_reverse[ch] = - m_beta[ch]/m_alpha[ch];
  }
  alpha_reverse.swap(m_alpha);
  beta_reverse.swap(m_beta);
  return true;
}

/**
 * In case that the normalization has been applied manually before copying
 * from cv::Mat data into El::Matrix<DataType>, clear the scaling parameters
 * to prevent them from being applied again during copying. In addition, store
 * the parameters used as to allow the manual reversal transform afterwards if
 * necessary.
 */
void cv_process::reset_normalization_params(void)
{
  m_alpha.swap(m_alpha_used);
  m_beta.swap(m_beta_used);
  m_alpha.clear();
  m_beta.clear();
}

/**
 * Calculates the linear transform parameters for normalization per-channel and
 * per-sample-image, save the parameters, and let the actual transform applied
 * during copying from cv::Mat data into El::Matrix<DataType> data to avoid
 * reading the image twice.
 */
bool cv_process::compute_normalization_params(const cv::Mat& image)
{ 
  return m_preprocessor.determine_normalization(image, m_alpha, m_beta);
}

/**
 * Calculates the linear transform parameters for normalization per-channel and
 * per-sample-image, and exports the parameters via alpha and beta. This does
 * not store the result internally.
 * @param _alpha The channel-wise scaling parameters for linear transform
 * @param _beta  The channel-wise shifting parameters for linear transform
 */
bool cv_process::compute_normalization_params(const cv::Mat& image,
  std::vector<double>& _alpha, std::vector<double>& _beta) const
{
  return m_preprocessor.determine_normalization(image, _alpha, _beta);
}

} // end of namespace lbann
#endif // __LIB_OPENCV
