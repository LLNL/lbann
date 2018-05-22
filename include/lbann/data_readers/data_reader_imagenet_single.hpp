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
// data_reader_imagenet_single .hpp .cpp - data reader class for ImageNet
//                                         dataset packed into a single file
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_SINGLE_HPP
#define LBANN_DATA_READER_IMAGENET_SINGLE_HPP

#include "data_reader_imagenet.hpp"
#include <vector>

namespace lbann {
class imagenet_reader_single : public imagenet_reader {
 public:
  imagenet_reader_single(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  imagenet_reader_single(const imagenet_reader_single& source);
  imagenet_reader_single& operator=(const imagenet_reader_single& source);
  ~imagenet_reader_single() override;

  imagenet_reader_single* copy() const override { return new imagenet_reader_single(*this); }

  std::string get_type() const override {
    return "imagenet_reader_single";
  }

  // ImageNet specific functions
  void load() override;

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) override;
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid) override;

 private:
  std::vector<std::ifstream*> m_data_filestream;
  size_t m_file_size;
  std::vector<std::vector<unsigned char> > m_work_buffer;
  std::vector<std::pair<size_t, int> > m_offsets; //stores: <offset, label>

  void open_data_stream();
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_IMAGENET_HPP
