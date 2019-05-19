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
// lbann_data_reader_imagenet_patches .hpp .cpp - extract patches from ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_PATCHES_HPP
#define LBANN_DATA_READER_IMAGENET_PATCHES_HPP

#include "data_reader_image.hpp"
#include "cv_process_patches.hpp"

namespace lbann {
class imagenet_reader_patches : public image_data_reader {
 public:
  imagenet_reader_patches(bool shuffle) = delete;
  imagenet_reader_patches(const std::shared_ptr<cv_process_patches>& pp, bool shuffle = true);
  imagenet_reader_patches(const imagenet_reader_patches&);
  imagenet_reader_patches& operator=(const imagenet_reader_patches&);
  ~imagenet_reader_patches() override;

  imagenet_reader_patches* copy() const override { return new imagenet_reader_patches(*this); }

  void setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) override;

  std::string get_type() const override {
    return "imagenet_reader_patches";
  }

  int get_linearized_data_size() const override {
    return m_image_linearized_size * m_num_patches;
  }
  const std::vector<int> get_data_dims() const override {
    return {m_num_patches*m_image_num_channels, m_image_height, m_image_width};
  }

 protected:
  void set_defaults() override;
  virtual bool replicate_processor(const cv_process_patches& pp, const int nthreads);
  virtual std::vector<CPUMat> create_datum_views(CPUMat& X, const int mb_idx) const;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;

 protected:
  int m_num_patches; ///< number of patches extracted
  /// preprocessor for patches duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process_patches> > m_pps;
  std::unique_ptr<cv_process_patches> m_master_pps;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_IMAGENET_PATCHES_HPP
