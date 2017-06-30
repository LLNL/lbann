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

#ifndef LBANN_CV_PROCESS_PATCHES_HPP
#define LBANN_CV_PROCESS_PATCHES_HPP

#include "cv_process.hpp"
#include "patchworks/patchworks_patch_descriptor.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

/// Similar to cv_process but works on patches that are extracted from an image
class cv_process_patches : public cv_process {
 protected:
  patchworks::patch_descriptor m_pd;
  bool m_self_label;

 public:
  cv_process_patches(void) : cv_process(), m_self_label(false) {}
  cv_process_patches(const bool self_label) : cv_process(), m_self_label(self_label) {}
  cv_process_patches(const cv_process_patches& rhs);
  cv_process_patches& operator=(const cv_process_patches& rhs);

  cv_process_patches(const cv_transform::cv_flipping flip_code, const bool tosplit)
    : cv_process(flip_code, tosplit) {}

  virtual ~cv_process_patches(void) {}

  void set_patch_descriptor(const patchworks::patch_descriptor& pd) {
    m_pd = pd;
  }
  patchworks::patch_descriptor& patch_descriptor(void) {
    return m_pd;
  }
  const patchworks::patch_descriptor& patch_descriptor(void) const {
    return m_pd;
  }
  bool is_self_labeling(void) const { return m_self_label; }
  virtual unsigned int get_patch_label(void) const { return m_pd.get_current_patch_idx(); }

  bool preprocess(const cv::Mat& image, std::vector<cv::Mat>& patches);
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_PROCESS_PATCHES_HPP
