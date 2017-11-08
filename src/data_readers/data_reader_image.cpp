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

#include "lbann/data_readers/data_reader_image.hpp"
#include <fstream>

namespace lbann {

image_data_reader::image_data_reader(bool shuffle)
  : generic_data_reader(shuffle),
    m_image_width(256),
    m_image_height(256),
    m_image_num_channels(3),
    m_num_labels(1000)
{}

image_data_reader::image_data_reader(const image_data_reader& rhs)
  : generic_data_reader(rhs),
    m_image_dir(rhs.m_image_dir),
    m_image_list(rhs.m_image_list),
    m_image_width(rhs.m_image_width),
    m_image_height(rhs.m_image_height),
    m_image_num_channels(rhs.m_image_num_channels),
    m_num_labels(rhs.m_num_labels)
{}

image_data_reader& image_data_reader::operator=(const image_data_reader& rhs) {
  generic_data_reader::operator=(rhs);
  m_image_dir = rhs.m_image_dir;
  m_image_list = rhs.m_image_list;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_num_labels = rhs.m_num_labels;

  return (*this);
}

void image_data_reader::set_input_params(const int width, const int height, const int num_ch, const int num_labels) {
  if ((width > 0) && (height > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
  } else if (!((width == 0) && (height == 0))) { // set but not valid
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader setup error: invalid input image sizes";
    throw lbann_exception(err.str());
  }
  if (num_ch > 0) {
    m_image_num_channels = num_ch;
  } else if (num_ch < 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader setup error: invalid number of channels of input images";
    throw lbann_exception(err.str());
  }
  if (num_labels > 0) {
    m_num_labels = num_labels;
  } else if (num_labels < 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader setup error: invalid number of labels";
    throw lbann_exception(err.str());
  }
}

bool image_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  const int label = m_image_list[data_id].second;
  Y.Set(label, mb_idx, 1);
  return true;
}

void image_data_reader::load() {
  //const std::string imageDir = get_file_dir();
  const std::string imageListFile = get_data_filename();

  m_image_list.clear();

  // load image list
  FILE *fplist = fopen(imageListFile.c_str(), "rt");
  if (!fplist) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: failed to open: " + imageListFile);
  }

  while (!feof(fplist)) {
    char imagepath[512];
    int imagelabel;
    if (fscanf(fplist, "%s%d", imagepath, &imagelabel) <= 1) {
      break;
    }
    m_image_list.push_back(std::make_pair(imagepath, imagelabel));
  }
  fclose(fplist);

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_image_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}

}  // namespace lbann
