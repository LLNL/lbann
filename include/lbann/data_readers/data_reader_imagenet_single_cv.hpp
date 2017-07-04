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
// lbann_data_reader_imagenet .hpp .cpp - generic_data_reader class for ImageNetSingle dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_SINGLE_CV_HPP
#define LBANN_DATA_READER_IMAGENET_SINGLE_CV_HPP

#include "data_reader_imagenet.hpp"
#include "image_preprocessor.hpp"
#include "cv_process.hpp"

namespace lbann {
class imagenet_readerSingle_cv : public imagenet_reader {
 public:
  imagenet_readerSingle_cv(int batchSize, std::shared_ptr<cv_process>& pp, bool shuffle = true);
  imagenet_readerSingle_cv(const imagenet_readerSingle_cv& source);
  ~imagenet_readerSingle_cv();

  imagenet_readerSingle_cv& operator=(const imagenet_readerSingle_cv& source);

  int fetch_data(Mat& X);
  int fetch_label(Mat& Y);
  void load();

 private:
  std::ifstream m_data_filestream;
  size_t m_file_size;
  std::vector<unsigned char> m_work_buffer;
  std::vector<std::pair<size_t, int> > m_offsets; //stores: <offset, label>
  //std::vector<unsigned char> m_pixels;
  std::shared_ptr<cv_process> m_pp;

  void open_data_stream();
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_IMAGENET_SINGLE_CV_HPP
