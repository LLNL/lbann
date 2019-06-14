////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
// mnist_reader .hpp .cpp - data reader class for MNIST dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_mnist.hpp"
#include "lbann/utils/file_utils.hpp"
#include <cstdio>

namespace lbann {

mnist_reader::mnist_reader(bool shuffle)
  : image_data_reader(shuffle) {
  set_defaults();
}

mnist_reader::mnist_reader()
  : mnist_reader(true) {}

void mnist_reader::set_defaults() {
  m_image_width = 28;
  m_image_height = 28;
  m_image_num_channels = 1;
  set_linearized_image_size();
  m_num_labels=10;
}

bool mnist_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  int pixelcount = m_image_width * m_image_height;
  std::vector<unsigned char>& tmp = m_image_data[data_id];

  for (int p = 0; p < pixelcount; p++) {
    X.Set(p, mb_idx, tmp[p+1]);
  }

  auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  augment(pixel_col, m_image_height, m_image_width, 1);
  normalize(pixel_col, 1);
  pixel_noise(pixel_col); //add noise to image, disable by default
  return true;
}

bool mnist_reader::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  if(!m_gan_labelling) { //default
    unsigned char label = m_image_data[data_id][0];
    Y.Set(label, mb_idx, 1);
  } else {
    if(m_gan_label_value) Y.Set(m_gan_label_value,mb_idx,1); //fake sample is set to 1; adversarial model
    else { //fake sample (second half of minibatch is set to 0;discriminator model
      //mb_idx < (m_mb_size/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
      mb_idx < (get_current_mini_batch_size()/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
    }
  }
  return true;
}

//===================================================

void load_mnist_data(const std::string imagepath, const std::string labelpath,
  const int m_first_n, std::vector<std::vector<unsigned char> >& m_image_data) {

  // read labels
  FILE *fplbl = fopen(labelpath.c_str(), "rb");
  if (!fplbl) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: load_mnist_data: failed to open file: " + labelpath);
  }

  int magicnum1, numitems1;
  fread(&magicnum1, 4, 1, fplbl);
  fread(&numitems1, 4, 1, fplbl);
  __swapEndianInt((unsigned int&)magicnum1);
  __swapEndianInt((unsigned int&)numitems1);

  // read images
  FILE *fpimg = fopen(imagepath.c_str(), "rb");
  if (!fpimg) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: load_mnist_data: failed to open file: " + imagepath);
  }

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
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: load_mnist_data: numitems1 != numitems2");
  }

  if (m_first_n > 0) {
    numitems1 = m_first_n > numitems1 ? numitems1 : m_first_n;
  }

  // set to array
  m_image_data.resize(numitems1);
  for (int n = 0; n < numitems1; n++) {
    m_image_data[n].resize(1+(imgwidth * imgheight));
    fread(&m_image_data[n][0], 1, 1, fplbl);
    fread(&m_image_data[n][1], imgwidth * imgheight, 1, fpimg);
  }
  fclose(fpimg);
  fclose(fplbl);
}

void mnist_reader::load() {
  if (is_master()) {
    std::cerr << "starting lbann::mnist_reader::load\n";
  }
  m_image_data.clear();

  if(m_gan_labelling) m_num_labels=2;

  const std::string FileDir = get_file_dir();
  const std::string ImageFile = get_data_filename();
  const std::string LabelFile = get_label_filename();

  // set filepath
  const std::string imagepath = FileDir + "/" + ImageFile;
  const std::string labelpath = FileDir + "/" + LabelFile;

  if (is_master()) {
    std::cerr << "read labels!\n";
  }

  load_mnist_data(imagepath, labelpath, m_first_n, m_image_data);

  if (m_first_n > 0) {
    set_use_percent(1.0);
    set_absolute_sample_count(0u);
  }

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_image_data.size());
  for (size_t n = 0; n < m_shuffled_indices.size(); n++) {
    m_shuffled_indices[n] = n;
  }
  if (is_master()) {
    std::cerr << "calling select_subset_of_data; m_shuffled_indices.size: " <<
      m_shuffled_indices.size() << std::endl;
  }
  select_subset_of_data();
}

}  // namespace lbann
