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
// lbann_data_reader_cifar10 .hpp .cpp - generic_data_reader class for CIFAR10 dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_cifar10.hpp"

using namespace std;
using namespace El;
using namespace lbann;


cifar10_reader::cifar10_reader(int batchSize, bool shuffle)
  : generic_data_reader(batchSize, shuffle),
    m_image_width(32), m_image_height(32), m_image_num_channels(3) {
}

cifar10_reader::cifar10_reader(const cifar10_reader& source)
  : generic_data_reader((const generic_data_reader&) source),
    m_data(source.m_data),
    m_image_width(source.m_image_width),
    m_image_height(source.m_image_height),
    m_image_num_channels(source.m_image_num_channels) {
}

cifar10_reader& cifar10_reader::operator=(const cifar10_reader& source) {
  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  generic_data_reader::operator=(source);
  m_image_width = source.m_image_width;
  m_image_height = source.m_image_height;
  m_image_num_channels = source.m_image_num_channels;
  m_data = source.m_data;

  return (*this);
}

cifar10_reader::~cifar10_reader(void) { }

void cifar10_reader::load(void) {
  stringstream err;

  //open data file
  string image_dir = get_file_dir();
  string filename = get_data_filename();
  stringstream b;
  b << image_dir << "/" << filename;
  if (is_master()) {
    cout << "opening: " << b.str() << endl;
  }
  ifstream in(b.str().c_str(), ios::binary);
  if (not in.good()) {
    err << __FILE__ << " " << __LINE__
        << " ::  failed to open " << b.str() << " for reading";
    throw lbann_exception(err.str());
  }

  //get number of images, with error checking
  int len = get_linearized_data_size() + 1;  //should be 3073
  in.seekg(0, in.end);
  streampos fs = in.tellg();
  in.seekg(0, in.beg);
  if (fs % len != 0) {
    err << __FILE__ << " " << __LINE__
        << " ::  fs % len != 0; fs: " << fs << " len: " << len;
    throw lbann_exception(err.str());
  }

  //reserve space for string images
  int num_images = fs / len;
  m_data.resize(num_images);
  for (size_t h=0; h<m_data.size(); h++) {
    m_data[h].resize(len);
  }

  //read in the images; each image is 1 byte, which is the
  //label (0-9), and 2072 pixels
  for (size_t h=0; h<m_data.size(); h++) {
    in.read((char *)&(m_data[h][0]), len);
  }
  in.close();

  m_shuffled_indices.resize(m_data.size());
  for (size_t n = 0; n < m_data.size(); n++) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}


int lbann::cifar10_reader::fetch_data(Mat& X) {
  stringstream err;

  if(!generic_data_reader::position_valid()) {
    err << __FILE__ << " " << __LINE__ << " :: lbann::imagenet_reader::fetch_data() - !generic_data_reader::position_valid()";
    throw lbann_exception(err.str());
  }

  int current_batch_size = getm_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+current_batch_size),
                               m_shuffled_indices.size());
  for (int n = m_current_pos; n < end_pos; ++n) {
    int k = n - m_current_pos;
    int idx = m_shuffled_indices[n];
    for (size_t p = 1; p<m_data[idx].size(); p++) {
      X.Set(p-1, k, m_data[idx][p]);
    }

    auto pixel_col = X(IR(0, X.Height()), IR(k, k + 1));
    augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
    normalize(pixel_col, m_image_num_channels);
  }

  return end_pos - m_current_pos;
}

int lbann::cifar10_reader::fetch_label(Mat& Y) {
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
    int label = (int)m_data[index][0];
    Y.Set(label, k, 1);
  }
  return (n - m_current_pos);
}
