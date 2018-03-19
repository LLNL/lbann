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
// data_reader_imagenet .hpp .cpp - data reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_store/data_store_imagenet.hpp"
#include <omp.h>

namespace lbann {

imagenet_reader::imagenet_reader(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : image_data_reader(shuffle) {
  set_defaults();

  if (!pp) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: " << get_type() << " construction error: no image processor";
    throw lbann_exception(err.str());
  }

  replicate_processor(*pp);
}

imagenet_reader::imagenet_reader(const imagenet_reader& rhs)
  : image_data_reader(rhs) {
  if (rhs.m_pps.size() == 0u || !rhs.m_pps[0]) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: " << get_type() << " construction error: no image processor";
    throw lbann_exception(err.str());
  }
  replicate_processor(*rhs.m_pps[0]);
}

imagenet_reader& imagenet_reader::operator=(const imagenet_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  image_data_reader::operator=(rhs);

  if (rhs.m_pps.size() == 0u || !rhs.m_pps[0]) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: " << get_type() << " construction error: no image processor";
    throw lbann_exception(err.str());
  }
  replicate_processor(*rhs.m_pps[0]);
  return (*this);
}

imagenet_reader::~imagenet_reader() {
}

void imagenet_reader::set_defaults() {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
}

/// Replicate image processor for each OpenMP thread
bool imagenet_reader::replicate_processor(const cv_process& pp) {
  const int nthreads = omp_get_max_threads();
  m_pps.resize(nthreads);

  // Construct thread private preprocessing objects out of a shared pointer
  #pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < nthreads; ++i) {
    //auto ppu = std::make_unique<cv_process>(pp); // c++14
    std::unique_ptr<cv_process> ppu(new cv_process(pp));
    m_pps[i] = std::move(ppu);
  }

  bool ok = true;
  for (int i = 0; ok && (i < nthreads); ++i) {
    if (!m_pps[i]) ok = false;
  }

  if (!ok || (nthreads <= 0)) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: " << get_type() << " cannot replicate image processor";
    throw lbann_exception(err.str());
    return false;
  }

  const std::vector<unsigned int> dims = pp.get_data_dims();
  if ((dims.size() == 2u) && (dims[0] != 0u) && (dims[1] != 0u)) {
    m_image_width = static_cast<int>(dims[0]);
    m_image_height = static_cast<int>(dims[1]);
    set_linearized_image_size();
  }

  return true;
}

::Mat imagenet_reader::create_datum_view(::Mat& X, const int mb_idx) const {
  return El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
}

bool imagenet_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  int width=0, height=0, img_type=0;

  std::vector<unsigned char> *image_buf;

  ::Mat X_v = create_datum_view(X, mb_idx);

  bool ret;
  if (m_data_store != nullptr) {
    m_data_store->get_data_buf(data_id, image_buf, tid);
    ret = lbann::image_utils::load_image(*image_buf, width, height, img_type, *(m_pps[tid]), X_v);
  } else {
    ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_pps[tid]), X_v);
  }

  if(!ret) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                          + get_type() + ": image_utils::load_image failed to load - "
                          + imagepath);
  }
  if((width * height * CV_MAT_CN(img_type)) != m_image_linearized_size) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                          + get_type() + ": mismatch data size -- either width, height or channel - "
                          + imagepath + "[w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height)
                          + "x" + std::to_string(CV_MAT_CN(img_type)) + "]");
  }
  return true;
}

}  // namespace lbann
