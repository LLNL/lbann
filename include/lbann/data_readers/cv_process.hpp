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
// cv_process .cpp .hpp - structure that defines the operations
//                        on image data in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_PROCESS_HPP
#define LBANN_CV_PROCESS_HPP

#include "cv_transform.hpp"
#include "cv_normalizer.hpp"
#include "cv_subtractor.hpp"
#include "cv_augmenter.hpp"
#include "cv_colorizer.hpp"
#include "cv_decolorizer.hpp"
#include "cv_cropper.hpp"
#include "cv_resizer.hpp"
#include "cv_mean_extractor.hpp"
#include <memory>
#include <limits> // std::numeric_limits

#ifdef LBANN_HAS_OPENCV
namespace lbann {

/** A structure packs the parameters for image pre-/post-processing that takes
 *  advantage of the OpenCV framework.
 */
class cv_process {
  /// OpenCV flip codes: c<0 for top_left <-> bottom_right, c=0 for top<->down, and c>0 for left<->right

 protected:
  /// unique name for the processor
  std::string m_name;
  /// Whether to flip an image
  cv_transform::cv_flipping m_flip;
  /// Whether to split channels
  bool m_split;
  /// whether a normalizing transform is set or not
  bool m_is_normalizer_set;
  /// The index of the normalizing transform in the array of transforms
  unsigned int m_normalizer_idx;

  /// Array of transforms
  std::vector<std::unique_ptr<cv_transform> > m_transforms;

  /// Check if the last transform registered in the list is a normalizer and not a subtractor
  bool to_fuse_normalizer_with_copy() const;

  void set_normalizer_info();

 public:
  cv_process()
    : m_flip(cv_transform::_no_flip_), m_split(true), m_is_normalizer_set(false), m_normalizer_idx(0u) {}

  cv_process(const cv_process& rhs);
  cv_process& operator=(const cv_process& rhs);

  cv_process(const cv_transform::cv_flipping flip_code, const bool tosplit)
    : m_flip(flip_code), m_split(tosplit), m_is_normalizer_set(false), m_normalizer_idx(0u) {}

  virtual ~cv_process() {}

  std::string get_name() const { return m_name; }
  void set_name(const std::string& name) { m_name = name; }

  /// Reset all the transforms
  void reset();

  /// Check whether to flip
  bool to_flip() const {
    return (m_flip != cv_transform::_no_flip_);
  }
  /// Tell how to flip
  int how_to_flip() const {
    return static_cast<int>(m_flip);
  }
  /**
   *  Set the flipping behavior. This is to deal with custom image format, which
   *  is not supported by OpenCV's builtin decoders and may impose different pixel
   *  coordinate system in its custom decoder.
   *  It is not to substitute for random flipping in augmentation.
   */
  void set_to_flip(const cv_transform::cv_flipping f) {
    m_flip = f;
  }
  /// Set to split channels
  bool to_split() const {
    return m_split;
  }

  /// Export transform operator of normalizer to allow lazy application
  std::vector<cv_normalizer::channel_trans_t> get_transform_normalize() const;
  /// Export transform operator of normalizer for a specific channel
  std::vector<cv_normalizer::channel_trans_t> get_transform_normalize(const unsigned int ch) const;

  /// Turn off normalizer. This is useful to make sure it off after potential lazy application
  void disable_lazy_normalizer();

  /// Turn off all transforms
  void disable_transforms();

  /// Add a tranform
  bool add_transform(std::unique_ptr<cv_transform> tr);

  /// Add a normalizing tranform
  bool add_normalizer(std::unique_ptr<cv_normalizer> tr);
  bool add_normalizer(std::unique_ptr<cv_subtractor> tr);

  /// Allow access to the list of transforms registered
  const std::vector<std::unique_ptr<cv_transform> >& get_transforms() const {
    return m_transforms;
  }

  /// Allow read-only access to a particular transform indexed by idx
  const cv_transform* get_transform(const unsigned int idx) const;

  /// Allow read-write access to a particular transform indexed by idx
  cv_transform* get_transform(const unsigned int idx);

  /// Retrun the number of transforms registered
  unsigned int get_num_transforms() const { return m_transforms.size(); }

  /** Return final image dimension {width, height} after all the transforms
   *  If a cropper is set, returns {crop_width, crop_height}. Otherwise, {0,0}.
   */
  std::vector<unsigned int> get_data_dims() const;

  void determine_inverse_lazy_normalization();

  /// Execute a range of transforms [tr_strart, tr_end) on the given image in order
  bool preprocess(cv::Mat& image, unsigned int tr_start = 0u,
                  unsigned int tr_end = std::numeric_limits<unsigned int>::max());
  /// Execute all the inverse transforms on the given image in the reverse order
  bool postprocess(cv::Mat& image);

  virtual std::string get_type() const { return "cv_process"; }
  virtual std::string get_description() const;
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_PROCESS_HPP
