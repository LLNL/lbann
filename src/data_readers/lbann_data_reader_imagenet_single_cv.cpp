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

#include "lbann/data_readers/lbann_data_reader_imagenet_single_cv.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <fstream>
using namespace std;
using namespace El;


lbann::imagenet_readerSingle_cv::imagenet_readerSingle_cv(int batchSize, std::shared_ptr<cv_process>& pp, bool shuffle)
  : imagenet_reader(batchSize, shuffle), m_pp(pp) {
  //m_pixels.resize(m_image_width * m_image_height * m_image_num_channels);
  if (!m_pp) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader construction error: invalid image processor";
    throw lbann_exception(err.str());
  }
}

lbann::imagenet_readerSingle_cv::imagenet_readerSingle_cv(const imagenet_readerSingle_cv& source)
  : imagenet_reader(source), m_pp(source.m_pp) {
  m_offsets = source.m_offsets;
  //m_pixels = source.m_pixels;
  open_data_stream();
}


lbann::imagenet_readerSingle_cv::~imagenet_readerSingle_cv(void) {
  m_data_filestream.close();
}


int lbann::imagenet_readerSingle_cv::fetch_label(Mat& Y) {
//@todo only one line is different from ImageNet:
//label = ... should be refactored to eliminate duplicate code
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
    int label = m_offsets[index+1].second;

    Y.Set(label, k, 1);
  }
  return (n - m_current_pos);
}

void lbann::imagenet_readerSingle_cv::load(void) {
  string image_dir = get_file_dir();
  string base_filename = get_data_filename();

  //open offsets file, with error checking
  stringstream b;
  b << image_dir << "/" << base_filename << "_offsets.txt";
  if (is_master()) {
    cout << "opening: " << b.str() << " " << endl;
  }
  ifstream in(b.str().c_str());
  if (not in.is_open() and in.good()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  failed to open " << b.str() << " for reading";
    throw lbann_exception(err.str());
  }

  //read the offsets file
  int num_images;
  in >> num_images;
  if (is_master()) {
    cout << "num images: " << num_images << endl;
  }
  m_offsets.reserve(num_images);
  m_offsets.push_back(make_pair(0,0));
  size_t last_offset = 0;
  size_t offset;
  int label;
  while (in >> offset >> label) {
    m_offsets.push_back(make_pair(offset + last_offset, label));
    last_offset = m_offsets.back().first;
  }

  if (num_images+1 != m_offsets.size()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  we read " << m_offsets.size() << " offsets, but should have read " << num_images;
    throw lbann_exception(err.str());
  }
  in.close();

  open_data_stream();

  m_shuffled_indices.resize(m_offsets.size());
  for (size_t n = 0; n < m_offsets.size()-1; n++) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}


int lbann::imagenet_readerSingle_cv::fetch_data(Mat& X) {
  stringstream err;

  if(!generic_data_reader::position_valid()) {
    err << __FILE__ << " " << __LINE__ << " :: lbann::imagenet_reader::fetch_data() - !generic_data_reader::position_valid()";
    throw lbann_exception(err.str());
  }

  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  int width, height;
  int current_batch_size = getm_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+current_batch_size), m_shuffled_indices.size());

  for (int n = m_current_pos; n < end_pos; ++n) {
    int k = n - m_current_pos;
    int idx = m_shuffled_indices[n];
    if (idx > m_offsets.size()-1) {
      err << __FILE__ << " " << __LINE__ << " :: idx= " << idx << " is larger than m_offsets.size()= " << m_offsets.size() << " -2";
      throw lbann_exception(err.str());
    }
    size_t start = m_offsets[idx].first;
    size_t end = m_offsets[idx+1].first;

    if (end > m_file_size) {
      err << __FILE__ << " " << __LINE__ << " :: end= " << end << " is larger than m_file_size= " << m_file_size << " for P_" << get_rank() << " with role: " << get_role() << " m_offsets.size(): " << m_offsets.size() << " n: " << n << " idx: " << idx;
      throw lbann_exception(err.str());
    }

    int ssz = end - start;

    if (ssz <= 0) {
      err << "P_" << get_rank() << " start: " << start << " end: " << end << " ssz= " << ssz << " is <= 0";
      throw lbann_exception(err.str());
    }

    m_work_buffer.resize(ssz);
    m_data_filestream.seekg(start);
    m_data_filestream.read((char *)&m_work_buffer[0], ssz);

    int img_type=0;
    ::Mat X_v;
    View(X_v, X, IR(0, X.Height()), IR(k, k + 1));
    bool ret = image_utils::import_image(m_work_buffer, width, height, img_type, *m_pp, X_v);

    if (_BUILTIN_FALSE(!ret)) {
      err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: image_utils::loadJPG failed to load index: " << idx;
      throw lbann_exception(err.str());
    }
    if (_BUILTIN_FALSE((width * height * CV_MAT_CN(img_type)) != num_channel_values)) {
      err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: mismatch data size -- either width or height";
      throw lbann_exception(err.str());
    }
  }

  return end_pos - m_current_pos;
}

// Assignment operator
lbann::imagenet_readerSingle_cv& lbann::imagenet_readerSingle_cv::operator=(const imagenet_readerSingle_cv& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  imagenet_reader::operator=(source);

  m_offsets = source.m_offsets;
  m_pp = source.m_pp;
  //m_pixels = source.m_pixels;
  open_data_stream();

  return (*this);
}

void lbann::imagenet_readerSingle_cv::open_data_stream(void) {
  string image_dir = get_file_dir();
  string base_filename = get_data_filename();
  stringstream b;
  b << image_dir << "/" << base_filename << "_data.bin";
  if (is_master()) {
    cout << "opening: " << b.str() << " " << endl;
  }
  m_data_filestream.open(b.str().c_str(), ios::in | ios::binary);
  if (not m_data_filestream.is_open() and m_data_filestream.good()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  failed to open " << b.str() << " for reading";
    throw lbann_exception(err.str());
  }
  m_data_filestream.seekg(0, m_data_filestream.end);
  m_file_size = m_data_filestream.tellg();
  m_data_filestream.seekg(0, m_data_filestream.beg);
}
