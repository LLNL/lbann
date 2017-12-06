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
// data_reader_image .hpp .cpp - generic data reader class for image dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_DATA_READER_HPP
#define IMAGE_DATA_READER_HPP

#include "data_reader.hpp"
#include "image_preprocessor.hpp"
#include "cv_process.hpp"

namespace lbann {
class image_data_reader : public generic_data_reader {
 public:
  image_data_reader(bool shuffle = true);
  image_data_reader(const image_data_reader&);
  image_data_reader& operator=(const image_data_reader&);

  /** Set up imagenet specific input parameters
   *  If argument is set to 0, then this method does not change the value of
   *  the corresponding parameter. However, width and height can only be both
   *  zero or both non-zero.
   */
  virtual void set_input_params(const int width=0, const int height=0, const int num_ch=0, const int num_labels=0);

  // dataset specific functions
  void load() override;

  int get_num_labels() const override {
    return m_num_labels;
  }
  virtual int get_image_width() const {
    return m_image_width;
  }
  virtual int get_image_height() const {
    return m_image_height;
  }
  virtual int get_image_num_channels() const {
    return m_image_num_channels;
  }
  int get_linearized_data_size() const override {
    return m_image_width * m_image_height * m_image_num_channels;
  }
  int get_linearized_label_size() const override {
    return m_num_labels;
  }
  const std::vector<int> get_data_dims() const override {
    return {m_image_num_channels, m_image_height, m_image_width};
  }

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) override {
    internal_save_image(pixels, filename, m_image_height, m_image_width,
                        m_image_num_channels, do_scale);
  }

 protected:
  /// Set the default values for the width, the height, the number of channels, and the number of labels of an image
  virtual void set_defaults();
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid) override;

 protected:
  std::string m_image_dir; ///< where images are stored
  std::vector<std::pair<std::string, int> > m_image_list; ///< list of image files and labels
  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels
  int m_num_labels; ///< number of labels
};

}  // namespace lbann

#endif  // IMAGE_DATA_READER_HPP
