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
// lbann_data_reader_mnist .hpp .cpp - DataReader class for MNIST dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include <stdio.h>

using namespace std;
using namespace El;

inline void __swapEndianInt(unsigned int& ui)
{
	ui = ((ui >> 24) | ((ui<<8) & 0x00FF0000) | ((ui>>8) & 0x0000FF00) | (ui << 24));
}



lbann::DataReader_MNIST::DataReader_MNIST(int batchSize, bool shuffle)
  : DataReader(batchSize, shuffle)
{
	ImageWidth = 28;
	ImageHeight = 28;
	NumLabels = 10;
}

lbann::DataReader_MNIST::DataReader_MNIST(int batchSize)
  : DataReader_MNIST(batchSize, true) {}

lbann::DataReader_MNIST::~DataReader_MNIST()
{
	this->free();
}

int lbann::DataReader_MNIST::fetch_data(Mat& X)
{
  if(!DataReader::position_valid()) {
    return 0;
  }

  int pixelcount = ImageWidth * ImageHeight;

  int n = 0;
  for (n = CurrentPos; n < CurrentPos + BatchSize; n++) {
    if (n >= (int)ShuffledIndices.size())
      break;

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    unsigned char* data = ImageData[index];
    unsigned char* pixels = &data[1];
    unsigned char label = data[0];

    for (int p = 0; p < pixelcount; p++) {
      X.Set(p, k, pixels[p] / 255.0f);
    }
  }

	return (n - CurrentPos);
}

int lbann::DataReader_MNIST::fetch_label(Mat& Y)
{
  if(!DataReader::position_valid()) {
    return 0;
  }

  int n = 0;
  for (n = CurrentPos; n < CurrentPos + BatchSize; n++) {
    if (n >= (int)ShuffledIndices.size())
      break;

    int k = n - CurrentPos;
    int index = ShuffledIndices[n];
    unsigned char* data = ImageData[index];
    unsigned char label = data[0];

    Y.Set(label, k, 1);
  }

	return (n - CurrentPos);
}

bool lbann::DataReader_MNIST::load(string FileDir, string ImageFile, string LabelFile, size_t max_sample_count, bool firstN)
{
	this->free();

  // set filepath
  string imagepath = FileDir + __DIR_DELIMITER + ImageFile;
  string labelpath = FileDir + __DIR_DELIMITER + LabelFile;

  // read labels
  FILE* fplbl = fopen(labelpath.c_str(), "rb");
  if (!fplbl)
    return false;

  int magicnum1, numitems1;
  fread(&magicnum1, 4, 1, fplbl);
  fread(&numitems1, 4, 1, fplbl);
  __swapEndianInt((unsigned int&)magicnum1);
  __swapEndianInt((unsigned int&)numitems1);

  // read images
  FILE* fpimg = fopen(imagepath.c_str(), "rb");
  if (!fpimg)
    return false;

  int magicnum2, numitems2, imgwidth, imgheight;
  fread(&magicnum2, 4, 1, fpimg);
  fread(&numitems2, 4, 1, fpimg);
  fread(&imgwidth, 4, 1, fpimg);
  fread(&imgheight, 4, 1, fpimg);
  __swapEndianInt((unsigned int&)magicnum2);
  __swapEndianInt((unsigned int&)numitems2);
  __swapEndianInt((unsigned int&)imgwidth);
  __swapEndianInt((unsigned int&)imgheight);

  if (numitems1 != numitems2) {
    fclose(fplbl);
    fclose(fpimg);
    return false;
  }

  // set to array
  for (int n = 0; n < numitems1; n++) {
    unsigned char* data = new unsigned char[1 + imgwidth * imgheight];
    fread(&data[0], 1, 1, fplbl);
    fread(&data[1], imgwidth * imgheight, 1, fpimg);

    ImageData.push_back(data);
  }
  fclose(fpimg);
  fclose(fplbl);

  // reset indices
  ShuffledIndices.clear();
  ShuffledIndices.resize(ImageData.size());
  //  ShuffledIndices.resize(80);
  for (size_t n = 0; n < ShuffledIndices.size(); n++) {
    ShuffledIndices[n] = n;
  }

  /// If the user requested fewer than the total data set size, select
  /// a random set from the entire data set.
	if (max_sample_count != 0) {
		max_sample_count = __MIN(max_sample_count, ImageData.size());
    if(!firstN) {
      std::shuffle(ShuffledIndices.begin(), ShuffledIndices.end(), get_generator());
    }
    ShuffledIndices.resize(max_sample_count);
    if(!firstN) {
      std::sort(ShuffledIndices.begin(), ShuffledIndices.end());
    }
  }

	return true;
}

void lbann::DataReader_MNIST::free()
{
  for (size_t n = 0; n < ImageData.size(); n++) {
    unsigned char* data = ImageData[n];
    delete [] data;
  }
  ImageData.clear();
}
