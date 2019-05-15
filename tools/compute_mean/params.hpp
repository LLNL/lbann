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
////////////////////////////////////////////////////////////////////////////////

#ifndef _TOOLS_COMPUTE_MEAN_CV_PARAMS_HPP_
#define _TOOLS_COMPUTE_MEAN_CV_PARAMS_HPP_
#include <utility>
#include <string>


namespace tools_compute_mean {

struct cropper_params {
  bool m_is_set;
  bool m_rand_center;
  std::pair<int, int> m_crop_sz;
  std::pair<int, int> m_roi_sz;

  cropper_params(void)
    : m_is_set(false),
      m_rand_center(false),
      m_crop_sz(std::make_pair(0, 0)),
      m_roi_sz(std::make_pair(0,0)) {}
};


class params {
 protected:
  /// Whether to enable cropper. The default is true.
  bool m_enable_cropper;
  /// Whether to write cropped result. The default is false.
  bool m_write_cropped;
  /// Whether to enable decolorizer. The default is false.
  bool m_enable_decolorizer;
  /// Whether to enable colorizer. The default is true.
  bool m_enable_colorizer;
  /// Whether to enable mean_extractor. The default is true.
  bool m_enable_mean_extractor;
  /// to only create output directories and do nothing else
  bool m_only_create_output_dirs;

  /**
   * The parameter used by mean_extractor. If 0, turns off mean_extractor.
   * The default is 1024.
   */
  unsigned int m_mean_batch_size;

  /**
   * The name of the data path file, which includes three paths.
   * 1. The root image data directory.
   * 2. The image data list, in which each line consists of a pair of the
   *    path of the image file relative to the root data directory and its label.
   * 3. The root output directory.
   */
  std::string m_data_path_file;

  /// File type extention used for writing cropped images. e.g. ".jpeg", ".png"
  std::string m_out_ext;

  /// The parameters used by cropper.
  cropper_params cp;

  /**
   * Controls the progress report frequency.
   * E.g., per every x percent at the root.
   */
  unsigned int report_freq;

  /// Seed for the random number generator
  int rng_seed;


 public:
  params(void)
    : m_enable_cropper(true),
      m_write_cropped(false),
      m_enable_decolorizer(false),
      m_enable_colorizer(true),
      m_enable_mean_extractor(true),
      m_mean_batch_size(1024u),
      m_out_ext(".png"),
      report_freq(10),
      rng_seed(42) {}

  bool set(int argc, char *argv[]);
  static std::string show_help(std::string name);

  bool to_enable_cropper() const {
    return m_enable_cropper;
  }
  bool to_write_cropped() const {
    return m_write_cropped;
  }
  bool to_enable_decolorizer() const {
    return m_enable_decolorizer;
  }
  bool to_enable_colorizer() const {
    return m_enable_colorizer;
  }
  bool to_enable_mean_extractor() const {
    return (m_enable_mean_extractor && (m_mean_batch_size > 0));
  }
  bool check_to_create_dirs_only() const {
    return m_only_create_output_dirs;
  }
  unsigned int get_mean_batch_size() const {
    return m_mean_batch_size;
  }
  std::string get_data_path_file() const {
    return m_data_path_file;
  }
  std::string get_out_ext() const {
    return m_out_ext;
  }
  void set_out_ext(const std::string ext) {
    m_out_ext = ext;
  }

  /// Allow read-only access to cropper parameters.
  const cropper_params& get_cropper_params() const {
    return cp;
  }
  /// Return the percent of progress to make to report once.
  float get_report_freq() const {
    const float f = report_freq/100.0;
    return ((f == 0.0 || f >= 1.0)? 1.0 : f);
  }
  /// Return the seed for random number generator
  int get_seed() const {
    return rng_seed;
  }
};

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_CV_PARAMS_HPP_
