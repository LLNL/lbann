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
// lbann_data_reader_cifar10 .hpp .cpp - generic_data_reader class for CIFAR10 dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_CIFAR10_HPP
#define LBANN_DATA_READER_CIFAR10_HPP

#include "data_reader_image.hpp"

namespace lbann {

class cifar10_reader : public image_data_reader {
 public:
  /// constructor
  cifar10_reader(bool shuffle = true);
  cifar10_reader(const cifar10_reader&) = default;
  cifar10_reader& operator=(const cifar10_reader&) = default;

  /// destructor
  ~cifar10_reader() override;

  cifar10_reader* copy() const override { return new cifar10_reader(*this); }

  std::string get_type() const override {
    return "cifar10_reader";
  }

  void set_input_params(const int, const int, const int, const int) override { set_defaults(); }
  void load() override;

 protected:
  void set_defaults() override;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

 private:
  std::vector<std::vector<unsigned char> > m_data;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CIFAR10_HPP
