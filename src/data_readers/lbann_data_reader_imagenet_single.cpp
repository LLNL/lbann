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
  : DataReader_ImageNet(batchSize, shuffle) 
{
  m_pixels.resize(m_image_width * m_image_height * m_image_num_channels);
}

lbann::DataReader_ImageNetSingle::DataReader_ImageNetSingle(const DataReader_ImageNetSingle& source)
  : DataReader_ImageNet(source) {}


lbann::DataReader_ImageNetSingle::~DataReader_ImageNetSingle() 
{
  m_data_filestream.close();
}

#if 0
int lbann::DataReader_ImageNetSingle::fetch_data(Mat& X)
{
  if(!DataReader::position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader load error: !position_valid";
    throw lbann_exception(err.str());
  }

  int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  int current_batch_size = getBatchSize();

  const int end_pos = Min(CurrentPos+current_batch_size, ShuffledIndices.size());
#pragma omp parallel for
  for (int n = CurrentPos; n < end_pos; ++n) {

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    string imagepath = m_image_dir + ImageList[index].first;

    int width, height;
    unsigned char* pixels = (unsigned char*) std::malloc(num_channel_values*sizeof(unsigned char));
    bool ret = lbann::image_utils::loadJPG(imagepath.c_str(), width, height, false, pixels);
    if(!ret) {
      throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
    }
    if(width != m_image_width || height != m_image_height) {
      throw lbann_exception("ImageNet: mismatch data size -- either width or height");
    }

    for (int p = 0; p < num_channel_values; p++) {
      X.Set(p, k, pixels[p]);
    }
    std::free(pixels);

    auto pixel_col = X(IR(0, X.Height()), IR(k, k + 1));
    augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
    normalize(pixel_col, m_image_num_channels);
  }

  return end_pos - CurrentPos;
}
#endif

int lbann::DataReader_ImageNetSingle::fetch_label(Mat& Y)
{
//@todo only one line if different from ImageNet: 
//label = ... should be refactored to eliminate duplicate code
  if(!position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader error: !position_valid";
    throw lbann_exception(err.str());
  }

  int current_batch_size = getBatchSize();
  int n = 0;
  for (n = CurrentPos; n < CurrentPos + current_batch_size; n++) {
    if (n >= (int)ShuffledIndices.size())
      break;

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    int label = m_offsets[index].second;

    Y.Set(label, k, 1);
  }
  return (n - CurrentPos);
}

void lbann::DataReader_ImageNetSingle::load()
{
  string image_dir = get_file_dir();
  string base_filename = get_data_filename();

  stringstream b;
  b << image_dir << "/" << base_filename << "_offsets.txt";
  ifstream in(b.str().c_str());
  if (not in.is_open() and in.good()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  failed to open " << b.str() << " for reading";
    throw lbann_exception(err.str());
  }

  size_t n;
  in >> n;
  m_offsets.reserve(n);
  m_offsets.push_back(make_pair(0,0));
  size_t offset;
  int label;
  while (in >> offset >> label) {
    m_offsets.push_back(make_pair(offset + m_offsets.back().first + 1, label));
  }
  
  if (n != m_offsets.size()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  we read " << m_offsets.size() << " offsets, but should have read " << n;
    throw lbann_exception(err.str());
  }
  in.close();

  b.clear();
  b.str("");
  b << image_dir << "/" << base_filename << "_data.bin";
  m_data_filestream.open(b.str().c_str(), ios::in | ios::binary);
  if (not m_data_filestream.is_open() and m_data_filestream.good()) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::  failed to open " << b.str() << " for reading";
    throw lbann_exception(err.str());
  }

  ShuffledIndices.resize(m_offsets.size());
  for (size_t n = 0; n < m_offsets.size(); n++) {
    ShuffledIndices[n] = n;
  }

  select_subset_of_data();
}


int lbann::DataReader_ImageNetSingle::fetch_data(Mat &X)
{
  stringstream err;

  if(!DataReader::position_valid()) {
    err << __FILE__ << " " << __LINE__ << " lbann::DataReader_ImageNet::fetch_data() - !DataReader::position_valid()";
    throw lbann_exception(err.str());
  }

  int width, height;
  int current_batch_size = getBatchSize();
  const int end_pos = Min(CurrentPos+current_batch_size, ShuffledIndices.size());
  for (int n = CurrentPos; n < end_pos; ++n) {
    int k = n - CurrentPos;
    int idx = ShuffledIndices[n];
    int start = m_offsets[idx].first;
    int end = m_offsets[idx+1].first;
    int ssz = end - start;
    if (ssz > m_work_buffer.size()) {
      m_work_buffer.resize(ssz);
    }
    char *w = &m_work_buffer[0];
    m_data_filestream.read(w, ssz);

    unsigned char *p = &m_pixels[0];
    bool ret = lbann::image_utils::loadJPG_buf(&m_work_buffer[0], width, height, false, p);

    if(!ret) {
      stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: ImageNetSingle: image_utils::loadJPG failed to load";
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
  }
  return 0;
}

// Assignment operator
lbann::DataReader_ImageNetSingle& lbann::DataReader_ImageNetSingle::operator=(const DataReader_ImageNetSingle& source)
{
  // check for self-assignment
  if (this == &source)
    return *this;

  // Call the parent operator= function
  DataReader_ImageNet::operator=(source);
}
