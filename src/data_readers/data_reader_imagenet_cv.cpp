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

#include "lbann/data_readers/data_reader_imagenet_cv.hpp"
#include "lbann/data_readers/image_utils.hpp"

#include <fstream>

namespace lbann {

imagenet_reader_cv::imagenet_reader_cv(int batchSize, std::shared_ptr<cv_process>& pp, bool shuffle)
  : generic_data_reader(batchSize, shuffle), m_pp(pp) {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  m_num_labels = 1000;

  if (!m_pp) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid image processor";
    throw lbann_exception(err.str());
  }
}

bool imagenet_reader_cv::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  int width=0, height=0, img_type=0;
  ::Mat X_v;

  El::View(X_v, X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));

  // Construct a thread private object out of a shared pointer
  cv_process pp(*m_pp);

  bool ret = lbann::image_utils::load_image(imagepath, width, height, img_type, pp, X_v);

  if(!ret) {
    throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
  }
  if ((width * height * CV_MAT_CN(img_type)) != num_channel_values) {
    throw lbann_exception("ImageNet: mismatch data size -- either width or height");
  }

  return true;
}

bool imagenet_reader_cv::fetch_datum(std::vector<::Mat>& X, int data_id, int mb_idx, int tid) {
  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  int width=0, height=0, img_type=0;
  std::vector<::Mat> X_v(X.size());

  for(unsigned int i=0u; i < X.size(); ++i) {
    El::View(X_v[i], X[i], El::IR(0, X[i].Height()), El::IR(mb_idx, mb_idx + 1));
  }

  // TODO: move this to an outer scope if possible to reduce overhead
  if (!std::dynamic_pointer_cast<cv_process_patches>(m_pp)) {
    throw lbann_exception("ImageNet: image_utils::invalid patch processor");
  }

  // Construct a thread private object out of a shared pointer
  cv_process_patches pp(*std::dynamic_pointer_cast<cv_process_patches>(m_pp));

  const bool ret = lbann::image_utils::load_image(imagepath, width, height, img_type, pp, X_v);

  if (pp.is_self_labeling()) {
    m_image_list[data_id].second = pp.get_patch_label();
  }

  if(!ret) {
    throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
  }
  if ((width * height * CV_MAT_CN(img_type)) != num_channel_values) {
    throw lbann_exception("ImageNet: mismatch data size -- either width or height");
  }

  return true;
}


int imagenet_reader_cv::fetch_data(std::vector<::Mat>& X) {
  const int nthreads = omp_get_max_threads();
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: imagenet data reader fetch error: !position_valid");
  }
  if (X.size() == 0u) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: imagenet data reader fetch error: X.size()==0");
  }
  const El::Int x_height = X[0].Height();
  const El::Int x_width = X[0].Width();
  for (auto & x : X) {
    El::Zeros(x, x_height, x_width);
  }

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  #pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nthreads; t++) {
    preprocess_data_source(omp_get_thread_num());
  }

  const int current_batch_size = getm_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+current_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    x_width);

  El::Zeros(m_indices_fetched_per_mb, mb_size, 1);
  #pragma omp parallel for
  for (int s = 0; s < mb_size; s++) {
    const int n = m_current_pos + (s * m_sample_stride);
    const int index = m_shuffled_indices[n];
    const bool valid = fetch_datum(X, index, s, omp_get_thread_num());
    if (!valid) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: imagenet data reader load error: datum not valid");
    }
    m_indices_fetched_per_mb.Set(s, 0, index);
  }

  /// Allow each thread to perform any postprocessing necessary on the
  /// data source prior to fetching data
  #pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nthreads; t++) {
    postprocess_data_source(omp_get_thread_num());
  }

  return mb_size;
}

bool imagenet_reader_cv::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  int label = m_image_list[data_id].second;
  Y.Set(label, mb_idx, 1);
  return true;
}

void imagenet_reader_cv::load() {
  std::string imageDir = get_file_dir();
  std::string imageListFile = get_data_filename();

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
