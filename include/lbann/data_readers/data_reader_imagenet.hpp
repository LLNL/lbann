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
// data_reader_imagenet .hpp .cpp - generic_data_reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_HPP
#define LBANN_DATA_READER_IMAGENET_HPP

#include "data_reader_image.hpp"
#include "image_preprocessor.hpp"

namespace lbann {
class imagenet_reader : public image_data_reader {
 public:
  imagenet_reader(bool shuffle = true);
  imagenet_reader(const imagenet_reader&) = default;
  imagenet_reader& operator=(const imagenet_reader&) = default;
  ~imagenet_reader() override {}

  imagenet_reader* copy() const override { return new imagenet_reader(*this); }

  /// Set up imagenet specific input parameters
  void set_input_params(const int width=256, const int height=256, const int num_ch=3, const int num_labels=1000) override;

 protected:
  void set_defaults() override;
  void allocate_pixel_bufs();
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid) override;

 protected:
  std::vector<std::vector<unsigned char>> m_pixel_bufs;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_IMAGENET_HPP
