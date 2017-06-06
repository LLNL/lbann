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
// lbann_cv_process_patches .cpp .hpp - structure that defines the operations
//                      on patches extracted from an image in the opencv format
////////////////////////////////////////////////////////////////////////////////


#include "lbann/data_readers/lbann_cv_process_patches.hpp"

#ifdef __LIB_OPENCV
namespace lbann
{

cv_process_patches::cv_process_patches(const cv_process_patches& rhs)
: cv_process(rhs), m_pd(rhs.m_pd)
{}

cv_process_patches& cv_process_patches::operator=(const cv_process_patches& rhs)
{
  if (this == &rhs) return (*this);
  cv_process::operator=(rhs);
  m_pd = rhs.m_pd;

  return (*this);
}

/**
 * Preprocess patches extracted from an image.
 * @return true if successful
 */
bool cv_process_patches::preprocess(const cv::Mat& image, std::vector<cv::Mat>& patches)
{
  bool ok = true;
  patches.clear();

  ok = m_pd.extract_patches(image, patches);

  for (size_t i=0u; ok && (i < patches.size()); ++i)
    ok = cv_process::preprocess(patches[i]);

  return ok;
}

} // end of namespace lbann
#endif // __LIB_OPENCV
