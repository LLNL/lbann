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

#include "lbann/data_readers/lbann_data_reader_imagenet.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <fstream>
using namespace std;
using namespace El;

lbann::DataReader_ImageNet::DataReader_ImageNet(int batchSize, bool shuffle)
  : DataReader(batchSize, shuffle)
{
  m_image_width = 256;
  m_image_height = 256;
  m_image_depth = 3;
  m_num_labels = 1000;
  m_scale = true;
  m_variance = false;
  m_mean = false;
  m_z_score = false;

  m_pixels = new unsigned char[m_image_width * m_image_height * m_image_depth];
}

lbann::DataReader_ImageNet::DataReader_ImageNet(const DataReader_ImageNet& source)
  : DataReader((const DataReader&) source),
    m_image_dir(source.m_image_dir), 
    ImageList(source.ImageList),
    m_image_width(source.m_image_width), 
    m_image_height(source.m_image_height), 
    m_image_depth(source.m_image_depth),  
    m_num_labels(source.m_num_labels),
    m_scale(source.m_scale), 
    m_variance(source.m_variance), 
    m_mean(source.m_mean), 
    m_z_score(source.m_z_score)
{
  m_pixels = new unsigned char[m_image_width * m_image_height * m_image_depth];
  memcpy(this->m_pixels, source.m_pixels, m_image_width * m_image_height * m_image_depth);
}

lbann::DataReader_ImageNet::~DataReader_ImageNet()
{
  delete [] m_pixels;
}

int lbann::DataReader_ImageNet::fetch_data(Mat& X)
{
  static bool testme = true;
  if(!DataReader::position_valid()) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader load error: !position_valid";
    throw lbann_exception(err.str());
  }

  if (m_z_score) {
    m_scale = false;
    m_mean = false;
    m_variance = false;
  }

  int pixelcount = m_image_width * m_image_height * m_image_depth;
  int current_batch_size = getBatchSize();

  //special handling for these types of normalization
  std::vector<float> pixels;
  bool special = false;
  if (m_mean or m_variance or m_z_score) {
    special = true;
    pixels.resize(pixelcount);
  }

  float scale = m_scale ? 255.0 : 1.0;

  int n = 0;
  for (n = CurrentPos; n < CurrentPos + current_batch_size; n++) {
    if (n >= (int)ShuffledIndices.size())
      break;

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    string imagepath = m_image_dir + ImageList[index].first;

    int width, height;
    bool ret = lbann::image_utils::loadJPG(imagepath.c_str(), width, height, true, m_pixels);
    if(!ret) {
      throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
    }
    if(width != m_image_width || height != m_image_height) {
      throw lbann_exception("ImageNet: mismatch data size -- either width or height");
    }

    if (not special) {
      for (int p = 0; p < pixelcount; p++) {
        X.Set(p, k, m_pixels[p] / scale);
      }
    }

    else {
      for (int p = 0; p < pixelcount; p++) {
        pixels[p] = m_pixels[p];
      }
      for (int x=0; x<3; x++) {
        standardize(pixels, x);
      }
      for (int p = 0; p < pixelcount; p++) {
        X.Set(p, k, pixels[p]);
      }
    }
  }

  return (n - CurrentPos);
}

int lbann::DataReader_ImageNet::fetch_label(Mat& Y)
{
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
    int label = ImageList[index].second;

    Y.Set(label, k, 1);
  }
  return (n - CurrentPos);
}

bool lbann::DataReader_ImageNet::load(string imageDir, string imageListFile)
{
  m_image_dir = imageDir; /// Store the primary path to the images for use on fetch
  ImageList.clear();

  // load image list
  FILE* fplist = fopen(imageListFile.c_str(), "rt");
  if (!fplist) {
    cerr << __FILE__ << " " << __LINE__ << "failed to open: " << imageListFile << endl;
  }

  while (!feof(fplist)) {
    char imagepath[512];
    int imagelabel;
    if (fscanf(fplist, "%s%d", imagepath, &imagelabel) <= 1)
      break;
    ImageList.push_back(make_pair(imagepath, imagelabel));
  }
  fclose(fplist);

  // reset indices
  ShuffledIndices.clear();
  ShuffledIndices.resize(ImageList.size());
  for (size_t n = 0; n < ImageList.size(); n++) {
    ShuffledIndices[n] = n;
  }

  return true;
}

bool lbann::DataReader_ImageNet::load(string imageDir, string imageListFile, size_t max_sample_count, bool firstN)
{
  bool load_successful = false;

  load_successful = load(imageDir, imageListFile);

  if(max_sample_count > getNumData() || ((long) max_sample_count) < 0) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: ImageNet: data reader load error: invalid number of samples selected";
    throw lbann_exception(err.str());
  }
  select_subset_of_data(max_sample_count, firstN);

  return load_successful;
}

bool lbann::DataReader_ImageNet::load(string imageDir, string imageListFile, double use_percentage, bool firstN)
{
  bool load_successful = false;

  load_successful = load(imageDir, imageListFile);

  size_t max_sample_count = rint(getNumData()*use_percentage);

  if(max_sample_count > getNumData() || ((long) max_sample_count) < 0) {
    stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: ImageNet: data reader load error: invalid number of samples selected";
    throw lbann_exception(err.str());
  }
  select_subset_of_data(max_sample_count, firstN);

  return load_successful;
}

void lbann::DataReader_ImageNet::free()
{
  delete [] m_pixels;
}

// Assignment operator
lbann::DataReader_ImageNet& lbann::DataReader_ImageNet::operator=(const DataReader_ImageNet& source)
{
  // check for self-assignment
  if (this == &source)
    return *this;

  // Call the parent operator= function
  DataReader::operator=(source);

  // first we need to deallocate any data that this data reader is holding!
  delete [] m_pixels;

  this->m_image_dir = source.m_image_dir;
  this->ImageList = source.ImageList;
  this->m_image_width = source.m_image_width;
  this->m_image_height = source.m_image_height;
  this->m_image_depth = source.m_image_height;
  this->m_num_labels = source.m_num_labels;

  this->m_scale = source.m_scale;
  this->m_variance = source.m_variance;
  this->m_mean = source.m_mean;
  this->m_z_score = source.m_z_score;

  m_pixels = new unsigned char[m_image_width * m_image_height * m_image_depth];
  memcpy(this->m_pixels, source.m_pixels, m_image_width * m_image_height * m_image_depth);

  return *this;
}


int lbann::DataReader_ImageNet::fetch_data(std::vector<std::vector<unsigned char> > &data, size_t max_to_process)
{
  stringstream err;

  if(!DataReader::position_valid()) {
    err << __FILE__ << " " << __LINE__ << " lbann::DataReader_ImageNet::fetch_data() - !DataReader::position_valid()";
    throw lbann_exception(err.str());
  }

  size_t num_to_process = max_to_process > 0 ? max_to_process : ImageList.size();

  data.clear();
  data.reserve(num_to_process);

  int pixelcount = m_image_width * m_image_height * m_image_depth;
  vector<unsigned char> pixels(pixelcount);
  unsigned char *v = &(pixels[0]);

  int width, height;
  for (size_t n = 0; n < num_to_process; n++) {
    string imagepath = m_image_dir + ImageList[n].first;
    if (n < 10) cout << "DataReader_ImageNet::fetch_data(); loading: " << imagepath << endl;
    bool ret = lbann::image_utils::loadJPG(imagepath.c_str(), width, height, true, v);
    if(!ret) {
      throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
    }
    if(width != m_image_width || height != m_image_height) {
      throw lbann_exception("ImageNet: mismatch data size -- either width or height");
    }

    data.push_back(pixels);
  }
  return 0;
}

void lbann::DataReader_ImageNet::standardize(std::vector<float> &pixels, int offset) {
    float pixelcount = m_image_width * m_image_height;
    if (m_z_score) {
      float x_sqr = 0;
      float mean = 0;
      for (size_t p = offset; p < pixels.size(); p+= 3) {
        mean += pixels[p];
        x_sqr += (pixels[p] * pixels[p]);
      }

      mean /= pixelcount;
      x_sqr /= pixelcount;
      float std_dev = x_sqr - (mean*mean);
      std_dev = sqrt(std_dev);
      for (size_t p = offset; p < pixels.size(); p+= 3) {
        pixels[p] = (pixels[p] - mean) / std_dev;
      }
    }

    else {

      // optionally scale to: [0,1]
      // formula is:  x_i - min(x) / max(x) - min(x)
      // but since min(x) = 0 and max(x) <= 255, we simply divide by 255
      if (m_scale) {
        for (size_t p = offset; p < pixels.size(); p+= 3) {
          pixels[p] /= 255.0;
        }
      }

      // optionally subtract the mean
      if (m_mean) {
        float mean = 0;
        for (size_t p = offset; p < pixels.size(); p+= 3) {
          mean += pixels[p];
        }
        mean /= pixelcount;
        for (size_t p = offset; p < pixels.size(); p+= 3) {
          pixels[p] -= mean;
        }
      }

      // optionally standardize to unit variance;
      // note: we need to recompute the mean and standard deviation,
      //       in case we've rescaled (above) using min-max scaling
      if (m_variance) {
        float x_sqr = 0;
        float mean = 0;
        for (size_t p = offset; p < pixels.size(); p+= 3) {
          mean += pixels[p];
          x_sqr += (pixels[p] * pixels[p]);
        }
        mean /= pixelcount;
        x_sqr /= pixelcount;
        float std_dev = x_sqr - (mean*mean);
        std_dev = sqrt(std_dev);

        for (size_t p = offset; p < pixels.size(); p+= 3) {
          pixels[p] /= std_dev;
        }
      }
    }
}

