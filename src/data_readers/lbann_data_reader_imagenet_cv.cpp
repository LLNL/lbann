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
// lbann_data_reader_imagenet_cv .hpp .cpp - DataReader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_imagenet_cv.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <fstream>
using namespace std;
using namespace El;

lbann::DataReader_ImageNet_cv::DataReader_ImageNet_cv(int batchSize, std::shared_ptr<cv_process>& pp, bool shuffle)
  : DataReader(batchSize, shuffle), m_pp(pp) {
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

lbann::DataReader_ImageNet_cv::DataReader_ImageNet_cv(const DataReader_ImageNet_cv& source)
  : DataReader((const DataReader&) source),
    m_image_dir(source.m_image_dir),
    ImageList(source.ImageList),
    m_image_width(source.m_image_width),
    m_image_height(source.m_image_height),
    m_image_num_channels(source.m_image_num_channels),
    m_num_labels(source.m_num_labels),
    m_pp(source.m_pp) {
}

lbann::DataReader_ImageNet_cv::~DataReader_ImageNet_cv() {
}

int lbann::DataReader_ImageNet_cv::fetch_data(Mat& X) {
  if(!DataReader::position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader load error: !position_valid";
    throw lbann_exception(err.str());
  }

  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  const int current_batch_size = getBatchSize();
  const int end_pos = Min(CurrentPos+current_batch_size, ShuffledIndices.size());

  #pragma omp parallel for
  for (int n = CurrentPos; n < end_pos; ++n) {

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    string imagepath = m_image_dir + ImageList[index].first;

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

  return end_pos - CurrentPos;
}

int lbann::DataReader_ImageNet_cv::fetch_label(Mat& Y) {
  if(!position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader error: !position_valid";
    throw lbann_exception(err.str());
  }

  int current_batch_size = getBatchSize();
  int n = 0;
  for (n = CurrentPos; n < CurrentPos + current_batch_size; n++) {
    if (n >= (int)ShuffledIndices.size()) {
      break;
    }

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    int label = ImageList[index].second;

    Y.Set(label, k, 1);
  }
  return (n - CurrentPos);
}

void lbann::DataReader_ImageNet_cv::load() {
  string imageDir = get_file_dir();
  string imageListFile = get_data_filename();

  m_image_dir = imageDir; /// Store the primary path to the images for use on fetch
  ImageList.clear();

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
    ImageList.push_back(make_pair(imagepath, imagelabel));
  }
  fclose(fplist);

  // reset indices
  ShuffledIndices.clear();
  ShuffledIndices.resize(ImageList.size());
  for (size_t n = 0; n < ImageList.size(); n++) {
    ShuffledIndices[n] = n;
  }

  select_subset_of_data();
}

void lbann::DataReader_ImageNet_cv::free() {
}

// Assignment operator
lbann::DataReader_ImageNet_cv& lbann::DataReader_ImageNet_cv::operator=(const DataReader_ImageNet_cv& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  DataReader::operator=(source);

  this->m_image_dir = source.m_image_dir;
  this->ImageList = source.ImageList;
  this->m_image_width = source.m_image_width;
  this->m_image_height = source.m_image_height;
  this->m_image_num_channels = source.m_image_num_channels;
  this->m_num_labels = source.m_num_labels;
  this->m_pp = source.m_pp;

  return *this;
}
