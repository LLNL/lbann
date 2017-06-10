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
// lbann_data_reader_imagenet .hpp .cpp - generic_data_reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_HPP
#define LBANN_DATA_READER_IMAGENET_HPP

#include "lbann_data_reader.hpp"
#include "lbann_image_preprocessor.hpp"

namespace lbann {
class imagenet_reader : public generic_data_reader {
 public:
  imagenet_reader(int batchSize, bool shuffle = true);
  imagenet_reader(const imagenet_reader& source);
  ~imagenet_reader();

  virtual int fetch_data(Mat& X);
  virtual int fetch_label(Mat& Y);

  int get_num_labels() {
    return m_num_labels;
  }

  // ImageNet specific functions
  virtual void load();
  void free();

  int get_image_width() {
    return m_image_width;
  }
  int get_image_height() {
    return m_image_height;
  }
  int get_image_num_channels() {
    return m_image_num_channels;
  }
  int get_linearized_data_size() {
    return m_image_width * m_image_height * m_image_num_channels;
  }
  int get_linearized_label_size() {
    return m_num_labels;
  }

  imagenet_reader& operator=(const imagenet_reader& source);

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) {
    internal_save_image(pixels, filename, m_image_height, m_image_width,
                        m_image_num_channels, do_scale);
  }

 protected:
  std::string m_image_dir; // where images are stored
  std::vector<std::pair<std::string, int> > image_list; // list of image files and labels
  int m_image_width; // image width
  int m_image_height; // image height
  int m_image_num_channels; // number of image channels
  int m_num_labels; // number of labels
  unsigned char *m_pixels;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_IMAGENET_HPP
