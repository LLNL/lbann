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
// lbann_data_reader_imagenet_cv .hpp .cpp - generic_data_reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_imagenet_cv.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <fstream>
using namespace std;
using namespace El;

lbann::imagenet_reader_cv::imagenet_reader_cv(int batchSize, std::shared_ptr<cv_process>& pp, bool shuffle)
  : generic_data_reader(batchSize, shuffle), m_pp(pp) {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  m_num_labels = 1000;

  if (!m_pp) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid image processor";
    throw lbann_exception(err.str());
  }
}

lbann::imagenet_reader_cv::imagenet_reader_cv(const imagenet_reader_cv& source)
  : generic_data_reader((const generic_data_reader&) source),
    m_image_dir(source.m_image_dir),
    image_list(source.image_list),
    m_image_width(source.m_image_width),
    m_image_height(source.m_image_height),
    m_image_num_channels(source.m_image_num_channels),
    m_num_labels(source.m_num_labels),
    m_pp(source.m_pp) {
}

lbann::imagenet_reader_cv::~imagenet_reader_cv() {
}

int lbann::imagenet_reader_cv::fetch_data(Mat& X) {
  if(!generic_data_reader::position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader load error: !position_valid";
    throw lbann_exception(err.str());
  }

  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  const int current_batch_size = getm_batch_size();
  const int end_pos = Min(m_current_pos+current_batch_size, m_shuffled_indices.size());

  #pragma omp parallel for
  for (int n = m_current_pos; n < end_pos; ++n) {

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];
    string imagepath = m_image_dir + image_list[index].first;

    int width=0, height=0, img_type=0;
    ::Mat X_v;
    View(X_v, X, IR(0, X.Height()), IR(k, k + 1));
    cv_process pp(*m_pp);
    bool ret = lbann::image_utils::load_image(imagepath, width, height, img_type, pp, X_v);
    if(!ret) {
      throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
    }
    if (_BUILTIN_FALSE((width * height * CV_MAT_CN(img_type)) != num_channel_values)) {
      throw lbann_exception("ImageNet: mismatch data size -- either width or height");
    }
  }

  return end_pos - m_current_pos;
}

int lbann::imagenet_reader_cv::fetch_label(Mat& Y) {
  if(!position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader error: !position_valid";
    throw lbann_exception(err.str());
  }

  int current_batch_size = getm_batch_size();
  int n = 0;
  for (n = m_current_pos; n < m_current_pos + current_batch_size; n++) {
    if (n >= (int)m_shuffled_indices.size()) {
      break;
    }

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];
    int label = image_list[index].second;

    Y.Set(label, k, 1);
  }
  return (n - m_current_pos);
}

void lbann::imagenet_reader_cv::load() {
  string imageDir = get_file_dir();
  string imageListFile = get_data_filename();

  m_image_dir = imageDir; /// Store the primary path to the images for use on fetch
  image_list.clear();

  // load image list
  FILE *fplist = fopen(imageListFile.c_str(), "rt");
  if (!fplist) {
    stringstream err;
    err << __FILE__ << " " << __LINE__ << "failed to open: " << imageListFile << endl;
    throw lbann_exception(err.str());
  }

  while (!feof(fplist)) {
    char imagepath[512];
    int imagelabel;
    if (fscanf(fplist, "%s%d", imagepath, &imagelabel) <= 1) {
      break;
    }
    image_list.push_back(make_pair(imagepath, imagelabel));
  }
  fclose(fplist);

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(image_list.size());
  for (size_t n = 0; n < image_list.size(); n++) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}

void lbann::imagenet_reader_cv::free() {
}

// Assignment operator
lbann::imagenet_reader_cv& lbann::imagenet_reader_cv::operator=(const imagenet_reader_cv& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  generic_data_reader::operator=(source);

  this->m_image_dir = source.m_image_dir;
  this->image_list = source.image_list;
  this->m_image_width = source.m_image_width;
  this->m_image_height = source.m_image_height;
  this->m_image_num_channels = source.m_image_num_channels;
  this->m_num_labels = source.m_num_labels;
  this->m_pp = source.m_pp;

  return *this;
}
