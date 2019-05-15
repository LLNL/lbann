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
// data_reader_nci .hpp .cpp - generic_data_reader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_NCI_HPP
#define LBANN_DATA_READER_NCI_HPP

#include "data_reader_csv.hpp"

#define NCI_HAS_HEADER

namespace lbann {

class data_reader_nci : public csv_reader {
 public:
  data_reader_nci(bool shuffle = true);
  data_reader_nci();
  data_reader_nci(const data_reader_nci& source) = default;
  data_reader_nci& operator=(const data_reader_nci& source) = default;
  ~data_reader_nci() override {}
  data_reader_nci* copy() const override { return new data_reader_nci(*this); }

  std::string get_type() const override {
    return "data_reader_nci";
  }

  // Todo: Support regression/get response.
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_NCI_HPP
