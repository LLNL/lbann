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
// data_reader_imagenet .hpp .cpp - data reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_HPP
#define LBANN_DATA_READER_IMAGENET_HPP

#include "data_reader_image.hpp"

namespace lbann {
class imagenet_reader : public image_data_reader {
 public:
  imagenet_reader(bool shuffle = true);
  imagenet_reader(const imagenet_reader&,
                  const std::vector<int>& ds_sample_move_list);
  imagenet_reader(const imagenet_reader&,
                  const std::vector<int>& ds_sample_move_list, std::string role);
  imagenet_reader(const imagenet_reader&) = default;
  imagenet_reader& operator=(const imagenet_reader&) = default;
  ~imagenet_reader() override;

  imagenet_reader* copy() const override { return new imagenet_reader(*this); }

  std::string get_type() const override {
    return "imagenet_reader";
  }

 protected:
  void set_defaults() override;
  virtual CPUMat create_datum_view(CPUMat& X, const int mb_idx) const;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_IMAGENET_HPP
