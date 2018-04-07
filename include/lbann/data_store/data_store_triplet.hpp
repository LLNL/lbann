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

#ifndef __DATA_STORE_TRIPLET_HPP__
#define __DATA_STORE_TRIPLET_HPP__

#include "lbann/data_store/data_store_multi_images.hpp"

namespace lbann {

/**
 * todo
 */

class data_store_triplet : public data_store_multi_images {
 public:

  //! ctor
  data_store_triplet(generic_data_reader *reader, model *m) :
    data_store_multi_images(reader, m) {
    set_name("data_store_triplet");
  }

  //! copy ctor
  data_store_triplet(const data_store_triplet&) = default;

  //! operator=
  data_store_triplet& operator=(const data_store_triplet&) = default;

  data_store_triplet * copy() const override { return new data_store_triplet(*this); }

  //! dtor
  ~data_store_triplet() override {};

  void setup() override;

 protected :

  std::vector<std::string> get_sample(size_t idx) const override;
};

}  // namespace lbann

#endif  // __DATA_STORE_TRIPLET_HPP__
