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
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_IMAGENET_HPP__
#define __DATA_STORE_IMAGENET_HPP__

#include "lbann/data_store/data_store_image.hpp"

namespace lbann {

/**
 * todo
 */

class data_store_imagenet : public data_store_image {
 public:

  //! ctor
  data_store_imagenet(generic_data_reader *reader, model *m) :
    data_store_image(reader, m) {}

  //! copy ctor
  data_store_imagenet(const data_store_imagenet&) = default;

  //! operator=
  data_store_imagenet& operator=(const data_store_imagenet&) = default;

  data_store_imagenet * copy() const override { return new data_store_imagenet(*this); }

  //! dtor
  ~data_store_imagenet() override {};

  void setup() override;

 protected :

  void get_file_sizes() override;

  /// for use during development and testing
  virtual void test_data();

  /// for use during development and testing
  void test_file_sizes();

  void read_files() override;
};

}  // namespace lbann

#endif  // __DATA_STORE_IMAGENET_HPP__
