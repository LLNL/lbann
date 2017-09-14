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

#include "lbann/data_readers/data_reader_imagenet_single.hpp"
#include "lbann/data_readers/image_utils.hpp"

#include <fstream>
using namespace std;
using namespace El;

namespace lbann {

imagenet_readerSingle::imagenet_readerSingle(int batchSize, bool shuffle)
  : imagenet_reader(batchSize, shuffle) {
}

imagenet_readerSingle::imagenet_readerSingle(const imagenet_readerSingle& source)
  : imagenet_reader(source) {
  m_offsets = source.m_offsets;
  open_data_stream();
}


imagenet_readerSingle::~imagenet_readerSingle() {
  m_data_filestream.close();
}


bool imagenet_readerSingle::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  int label = m_offsets[data_id+1].second;
  Y.Set(label, mb_idx, 1);
  return true;
}

void imagenet_readerSingle::load() {
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
  size_t num_images;
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


bool imagenet_readerSingle::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  stringstream err;
  int width, height;

  if (static_cast<size_t>(data_id+1) >= m_offsets.size()) {
    err << __FILE__ << " " << __LINE__ << " :: data_id= " << data_id << " is larger than m_offsets.size()= " << m_offsets.size() << " -2";
    throw lbann_exception(err.str());
  }
  size_t start = m_offsets[data_id].first;
  size_t end = m_offsets[data_id+1].first;

  if (end > m_file_size) {
    err << __FILE__ << " " << __LINE__ << " :: end= " << end << " is larger than m_file_size= " << m_file_size << " for P_" << get_rank() << " with role: " << get_role() << " m_offsets.size(): " << m_offsets.size() << " mb_idx: " << mb_idx << " data_id: " << data_id;
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

  unsigned char *pixels = m_pixel_bufs[tid].data();
  bool ret = lbann::image_utils::loadJPG(m_work_buffer, width, height, false, pixels);

  if(!ret) {
    err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: image_utils::loadJPG failed to load index: " << data_id;
    throw lbann_exception(err.str());
  }
  if(width != m_image_width || height != m_image_height) {
    err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: mismatch data size -- either width or height";
    throw lbann_exception(err.str());
  }

  for (El::Int p = 0; p < num_channel_values; p++) {
    X.Set(p, mb_idx, pixels[p]);
  }

  auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
  normalize(pixel_col, m_image_num_channels);

  return true;
}

// Assignment operator
imagenet_readerSingle& imagenet_readerSingle::operator=(const imagenet_readerSingle& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  imagenet_reader::operator=(source);

  m_offsets = source.m_offsets;
  open_data_stream();

  return (*this);
}

void imagenet_readerSingle::open_data_stream() {
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

}  // namespace lbann
