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

#ifndef LBANN_CV_PROCESS_PATCHES_HPP
#define LBANN_CV_PROCESS_PATCHES_HPP

#include "cv_process.hpp"
#include "patchworks/patchworks_patch_descriptor.hpp"
#include <limits> // std::numeric_limits

#ifdef LBANN_HAS_OPENCV
namespace lbann {

/// Similar to cv_process but works on patches that are extracted from an image
class cv_process_patches : public cv_process {
 protected:
  patchworks::patch_descriptor m_pd;
  bool m_self_label;
  unsigned int m_when_to_extract;

 public:
  cv_process_patches();
  cv_process_patches(const bool self_label);
  cv_process_patches(const cv_process_patches& rhs);
  cv_process_patches(const cv_transform::cv_flipping flip_code, const bool tosplit);
  cv_process_patches& operator=(const cv_process_patches& rhs);

  ~cv_process_patches() override {}

  void set_patch_descriptor(const patchworks::patch_descriptor& pd,
                            const unsigned int when_to_extract =
                                  std::numeric_limits<unsigned int>::max());
  patchworks::patch_descriptor& patch_descriptor() {
    return m_pd;
  }
  const patchworks::patch_descriptor& patch_descriptor() const {
    return m_pd;
  }
  unsigned int get_when_to_extract() const { return m_when_to_extract; }
  bool is_self_labeling() const { return m_self_label; }
  unsigned int get_num_labels() const { return m_pd.get_num_labels(); }
  virtual unsigned int get_patch_label() const { return m_pd.get_last_label(); }
  unsigned int get_num_patches() const { return m_pd.get_num_patches(); }
  std::vector<unsigned int> get_data_dims() const {
    return {m_pd.get_num_patches(), m_pd.get_patch_width(), m_pd.get_patch_height()};
  }

  bool preprocess(cv::Mat& image, std::vector<cv::Mat>& patches);

  std::string get_type() const override { return "cv_process_patches"; }
  std::string get_description() const override;
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_PROCESS_PATCHES_HPP
