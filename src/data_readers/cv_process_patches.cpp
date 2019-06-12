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
// cv_process_patches .cpp .hpp - structure that defines the operations
//                      on patches extracted from an image in the opencv format
////////////////////////////////////////////////////////////////////////////////


#include "lbann/data_readers/cv_process_patches.hpp"
#include <limits> // std::numeric_limits

#ifdef LBANN_HAS_OPENCV
namespace lbann {

cv_process_patches::cv_process_patches()
  : cv_process(), m_self_label(false),
    m_when_to_extract(std::numeric_limits<unsigned int>::max()) {
}

cv_process_patches::cv_process_patches(const bool self_label)
  : cv_process(), m_self_label(self_label),
    m_when_to_extract(std::numeric_limits<unsigned int>::max()) {
}

cv_process_patches::cv_process_patches(const cv_process_patches& rhs)
  : cv_process(rhs), m_pd(rhs.m_pd), m_self_label(rhs.m_self_label),
    m_when_to_extract(rhs.m_when_to_extract) {
}

cv_process_patches::cv_process_patches(const cv_transform::cv_flipping flip_code, const bool tosplit)
  : cv_process(flip_code, tosplit), m_self_label(false),
    m_when_to_extract(std::numeric_limits<unsigned int>::max()) {
}

cv_process_patches& cv_process_patches::operator=(const cv_process_patches& rhs) {
  if (this == &rhs) {
    return (*this);
  }
  cv_process::operator=(rhs);
  m_pd = rhs.m_pd;
  m_self_label = rhs.m_self_label;
  m_when_to_extract = rhs.m_when_to_extract;

  return (*this);
}

void cv_process_patches::set_patch_descriptor(const patchworks::patch_descriptor& pd,
                                              const unsigned int when_to_extract) {
  m_pd = pd;
  m_self_label = m_pd.is_self_labeling();
  m_when_to_extract = when_to_extract;
}

/**
 * Preprocess patches extracted from an image.
 * @return true if successful
 */
bool cv_process_patches::preprocess(cv::Mat& image, std::vector<cv::Mat>& patches) {
  bool ok = true;
  patches.clear();

  ok = cv_process::preprocess(image, 0u, m_when_to_extract);
  ok = ok && m_pd.extract_patches(image, patches);

  for (size_t i=0u; ok && (i < patches.size()); ++i) {
    ok = cv_process::preprocess(patches[i], m_when_to_extract);
  }

  return ok;
}

std::string cv_process_patches::get_description() const {
  std::stringstream os;
  const unsigned int when_to_extract = ((m_when_to_extract > m_transforms.size())?
                                         m_transforms.size() : m_when_to_extract);
  const std::string when_exactly = ((when_to_extract == 0u)?
    "at the beginning" : ("after " + m_transforms[when_to_extract-1]->get_name()));
  os << cv_process::get_description();
  os << " - self-labeling: " << m_self_label << std::endl
     << " - extract patches " << when_exactly << std::endl
     << m_pd  << std::endl;

  return os.str();
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
