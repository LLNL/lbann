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
// data_reader_multi_images .hpp .cpp - generic data reader class for datasets
//                                      employing multiple images per sample
////////////////////////////////////////////////////////////////////////////////

#ifndef DATA_READER_MULTI_IMAGES_HPP
#define DATA_READER_MULTI_IMAGES_HPP

#include "data_reader_imagenet.hpp"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

namespace lbann {
class data_reader_multi_images : public imagenet_reader {
 public:
  using img_src_t = std::vector<std::string>;
  using sample_t = std::pair<img_src_t, label_t>;

  data_reader_multi_images(bool shuffle = true);
  data_reader_multi_images(const data_reader_multi_images&);
  data_reader_multi_images& operator=(const data_reader_multi_images&);
  ~data_reader_multi_images() override;

  data_reader_multi_images* copy() const override {
    return new data_reader_multi_images(*this);
  }

  std::string get_type() const override {
    return "data_reader_multi_images";
  }

  /** Set up imagenet specific input parameters
   *  If argument is set to 0, then this method does not change the value of
   *  the corresponding parameter. However, width and height can only be both
   *  zero or both non-zero.
   */
  void set_input_params(const int width, const int height, const int num_ch,
                        const int num_labels, const int num_img_srcs);

  void set_input_params(const int width, const int height, const int num_ch,
                        const int num_labels) override;

  // dataset specific functions
  void load() override;

  int get_linearized_data_size() const override {
    return m_image_linearized_size * m_num_img_srcs;
  }
  const std::vector<int> get_data_dims() const override {
    return {static_cast<int>(m_num_img_srcs)*m_image_num_channels, m_image_height, m_image_width};
  }

  /// Return the sample list of current minibatch
  std::vector<sample_t> get_image_list_of_current_mb() const;

  /// Allow read-only access to the entire sample list
  const std::vector<sample_t>& get_image_list() const {
    return m_image_list;
  }

  sample_t get_sample(size_t idx) const {
    return m_image_list.at(idx);
  }

  /// The number of image sources or the number of siamese heads. e.g., 2;
  /// this method is added to support data_store functionality
  unsigned int get_num_img_srcs() const {
    return m_num_img_srcs;
  }

 protected:
  void set_defaults() override;
  virtual std::vector<CPUMat> create_datum_views(CPUMat& X, const int mb_idx) const;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

  bool read_text_stream(std::istream& text_stream, std::vector<sample_t>& list);
  bool load_list(const std::string file_name, std::vector<sample_t>& list,
                 const bool fetch_list_at_once = false);

 protected:
  std::vector<sample_t> m_image_list; ///< list of image files and labels
  /// The number of image sources or the number of siamese heads. e.g., 2
  unsigned int m_num_img_srcs;
};

}  // namespace lbann

#endif  // DATA_READER_MULTI_IMAGES_HPP
