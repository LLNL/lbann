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
#include <omp.h>

namespace lbann {

imagenet_reader_cv::imagenet_reader_cv(int batchSize, const std::shared_ptr<cv_process>& pp, bool shuffle)
  : generic_data_reader(batchSize, shuffle) {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  m_num_labels = 1000;

  const bool ok = replicate_preprocessor(pp);
  if (!ok) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid image processor";
    throw lbann_exception(err.str());
  }
}

/// Replicate preprocessor for each OpenMP thread
bool imagenet_reader_cv::replicate_preprocessor(const std::shared_ptr<cv_process>& pp) {
  if (!pp) return false;

  const int nthreads = omp_get_max_threads();

  // Construct thread private preprocessing objects out of a shared pointer
  if (std::dynamic_pointer_cast<cv_process_patches>(pp)) {
    m_pps.clear();
    m_ppps.resize(nthreads);
    const std::shared_ptr<const cv_process_patches> ppp = std::dynamic_pointer_cast<cv_process_patches>(pp);

    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < nthreads; ++i) {
      std::unique_ptr<cv_process_patches> ppu(new cv_process_patches(*ppp));
      m_ppps[i] = std::move(ppu);
    }

    for (int i = 0; i < nthreads; ++i)
      if (!m_ppps[i]) return false;
  } else {
    m_pps.resize(nthreads);
    m_ppps.clear();

    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < nthreads; ++i) {
      std::unique_ptr<cv_process> ppu(new cv_process(*pp));
      m_pps[i] = std::move(ppu);
    }

    for (int i = 0; i < nthreads; ++i)
      if (!m_pps[i]) return false;
  }

  if (nthreads <= 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid nthreads";
    throw lbann_exception(err.str());
  }

  return true;
}

imagenet_reader_cv::imagenet_reader_cv(const imagenet_reader_cv& rhs)
  : generic_data_reader(rhs),
    m_image_dir(rhs.m_image_dir),
    m_image_list(rhs.m_image_list),
    m_image_width(rhs.m_image_width),
    m_image_height(rhs.m_image_height),
    m_image_num_channels(rhs.m_image_num_channels),
    m_num_labels(rhs.m_num_labels)
{
  const bool ok = (rhs.m_pps.size() == 0u) ||
                  replicate_preprocessor(std::make_shared<cv_process>(*rhs.m_pps[0]));
  if (!ok) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid image processor";
    throw lbann_exception(err.str());
  }
}

imagenet_reader_cv& imagenet_reader_cv::operator=(const imagenet_reader_cv& rhs) {
  generic_data_reader::operator=(rhs);
  m_image_dir = rhs.m_image_dir;
  m_image_list = rhs.m_image_list;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_num_labels = rhs.m_num_labels;
  const bool ok = (rhs.m_pps.size() == 0u) ||
                  replicate_preprocessor(std::make_shared<cv_process>(*rhs.m_pps[0]));
  if (!ok) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid image processor";
    throw lbann_exception(err.str());
  }
  return (*this);
}

imagenet_reader_cv::~imagenet_reader_cv() {
}

bool imagenet_reader_cv::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  int width=0, height=0, img_type=0;
  ::Mat X_v;

  El::View(X_v, X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));

  const bool ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_pps[tid]), X_v);

  if(!ret) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "ImageNet: image_utils::load_image failed to load - " 
                          + imagepath);
  }
  if((width * height * CV_MAT_CN(img_type)) != num_channel_values) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "ImageNet: mismatch data size -- either width, height or channel - "
                          + imagepath + "[w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height)
                          + "x" + std::to_string(CV_MAT_CN(img_type)) + "]");
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

/*
  if (!m_ppps[tid]) { // fetch_data() and replicate_preprocessor() will check
    throw lbann_exception("ImageNet: imagenet_reader_cv::fetch_datum invalid patch processor");
  }
*/

  const bool ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_ppps[tid]), X_v);

  if (m_ppps[tid]->is_self_labeling()) {
    m_image_list[data_id].second = m_ppps[tid]->get_patch_label();
  }

  if(!ret) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "ImageNet: image_utils::load_image failed to load - " 
                          + imagepath);
  }
  if((width * height * CV_MAT_CN(img_type)) != num_channel_values) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "ImageNet: mismatch data size -- either width, height or channel - "
                          + imagepath + "[w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height)
                          + "x" + std::to_string(CV_MAT_CN(img_type)) + "]");
  }
  return true;
}

int imagenet_reader_cv::fetch_data(std::vector<::Mat>& X) {
  if (m_ppps.size() == 0u) {
    // TODO: If the patch handling is separated into a child class, this check is not necessary
    throw lbann_exception("ImageNet: imagenet_reader_cv::fetch_data invalid patch processor");
  }
  const int nthreads = omp_get_max_threads();
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: generic data reader load error: !position_valid"
      + " -- current pos = " + std::to_string(m_current_pos)
      + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
  }
  if (X.size() == 0u) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: imagenet data reader fetch error: X.size()==0");
  }
  const El::Int x_height = X[0].Height();
  const El::Int x_width = X[0].Width();

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  #pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nthreads; t++) {
    preprocess_data_source(omp_get_thread_num());
  }

  const int current_batch_size = get_current_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+current_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    x_width);

  for (auto & x : X) {
    El::Zeros(x, x_height, x_width);
  }
  El::Zeros(m_indices_fetched_per_mb, mb_size, 1);
  #pragma omp parallel for schedule(static, 1)
  for (int s = 0; s < mb_size; s++) {
    // Catch exceptions within the OpenMP thread.
    try {
      const int n = m_current_pos + (s * m_sample_stride);
      const int index = m_shuffled_indices[n];
      fetch_datum(X, index, s, omp_get_thread_num());
      m_indices_fetched_per_mb.Set(s, 0, index);
    } catch (lbann_exception& e) {
      lbann_report_exception(e);
    } catch (exception& e) {
      El::ReportException(e);
    }
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
