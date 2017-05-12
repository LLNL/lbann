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
// lbann_cvimage_process_params .cpp .hpp - Image prerpocessing functions
////////////////////////////////////////////////////////////////////////////////


#include "lbann/data_readers/lbann_cv_process.hpp"

#ifdef __LIB_OPENCV
namespace lbann
{

bool cv_process::set_normalization_params(const std::vector<double>& a, const std::vector<double>& b)
{
  if (a.size() != b.size()) return false;
  m_alpha = a;
  m_beta = b;
  return true;
}

bool cv_process::compute_normalization_params(const cv::Mat& image)
{ 
  return m_preprocessor.determine_normalization(image, m_alpha, m_beta);
}

bool cv_process::compute_normalization_params(const cv::Mat& image,
  std::vector<double>& alpha, std::vector<double>& beta) const
{
  return m_preprocessor.determine_normalization(image, alpha, beta);
}

} // end of namespace lbann
#endif // __LIB_OPENCV
