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
// lbann_data_reader_imagenet .hpp .cpp - DataReader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_imagenet_single.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <fstream>
using namespace std;
using namespace El;


lbann::DataReader_ImageNetSingle::DataReader_ImageNetSingle(int batchSize, bool shuffle)
  : DataReader_ImageNet(batchSize, shuffle) {
  m_pixels.resize(m_image_width * m_image_height * m_image_num_channels);
}

lbann::DataReader_ImageNetSingle::DataReader_ImageNetSingle(const DataReader_ImageNetSingle& source)
  : DataReader_ImageNet(source) {
  m_offsets = source.m_offsets;
  m_pixels = source.m_pixels;
  openDataStream();
}


lbann::DataReader_ImageNetSingle::~DataReader_ImageNetSingle() {
  m_data_filestream.close();
}


int lbann::DataReader_ImageNetSingle::fetch_label(Mat& Y) {
//@todo only one line is different from ImageNet:
//label = ... should be refactored to eliminate duplicate code
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
    int label = m_offsets[index+1].second;

    Y.Set(label, k, 1);
  }
  return (n - CurrentPos);
}

void lbann::DataReader_ImageNetSingle::load() {
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
  int n;
  in >> n;
  if (is_master()) {
    cout << "num images: " << n << endl;
  }
  m_offsets.reserve(n);
  m_offsets.push_back(make_pair(0,0));
  size_t last_offset = 0;
  size_t offset;
  int label;
  while (in >> offset >> label) {
    m_offsets.push_back(make_pair(offset + last_offset, label));
    last_offset = m_offsets.back().first;
  }

  if (n+1 != m_offsets.size()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  we read " << m_offsets.size() << " offsets, but should have read " << n;
    throw lbann_exception(err.str());
  }
  in.close();

  openDataStream();

  ShuffledIndices.resize(m_offsets.size());
  for (size_t n = 0; n < m_offsets.size()-1; n++) {
    ShuffledIndices[n] = n;
  }

  select_subset_of_data();
}


int lbann::DataReader_ImageNetSingle::fetch_data(Mat& X) {
  stringstream err;

  if(!DataReader::position_valid()) {
    err << __FILE__ << " " << __LINE__ << " :: lbann::DataReader_ImageNet::fetch_data() - !DataReader::position_valid()";
    throw lbann_exception(err.str());
  }

  int width, height;
  int current_batch_size = getBatchSize();
  const int end_pos = Min(CurrentPos+current_batch_size, ShuffledIndices.size());

  for (int n = CurrentPos; n < end_pos; ++n) {
    int k = n - CurrentPos;
    int idx = ShuffledIndices[n];
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

    unsigned char *p = &m_pixels[0];
    bool ret = lbann::image_utils::loadJPG(m_work_buffer, width, height, false, p);

    if(!ret) {
      stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: image_utils::loadJPG failed to load index: " << idx;
      throw lbann_exception(err.str());
    }
    if(width != m_image_width || height != m_image_height) {
      stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: mismatch data size -- either width or height";
      throw lbann_exception(err.str());
    }

    for (size_t p = 0; p < m_pixels.size(); p++) {
      X.Set(p, k, m_pixels[p]);
    }

    auto pixel_col = X(IR(0, X.Height()), IR(k, k + 1));
    augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
    normalize(pixel_col, m_image_num_channels);
  }

  return end_pos - CurrentPos;
}

// Assignment operator
lbann::DataReader_ImageNetSingle& lbann::DataReader_ImageNetSingle::operator=(const DataReader_ImageNetSingle& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  DataReader_ImageNet::operator=(source);

  m_offsets = source.m_offsets;
  m_pixels = source.m_pixels;
  openDataStream();

  return (*this);
}

void lbann::DataReader_ImageNetSingle::openDataStream() {
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
