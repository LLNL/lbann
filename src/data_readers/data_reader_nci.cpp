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
// lbann_data_reader_nci .hpp .cpp - generic_data_reader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_nci.hpp"
#include <cstdio>
#include <string>
#include <omp.h>

namespace lbann {

data_reader_nci::data_reader_nci(bool shuffle)
  : csv_reader(shuffle) {
  set_response_col(2);
  enable_responses();
  set_label_col(3);
  set_separator(' ');
  // First five columns are metadata, not the sample.
  set_skip_cols(5);
  // Header is broken, so skip it.
  set_skip_rows(1);
  set_has_header(false);
  // Transform to binary classification.
  set_label_transform(
    [] (const std::string& s) -> int {
      return s == "rs" ? 0 : 1;
    });
}

data_reader_nci::data_reader_nci()
  : data_reader_nci(true) {}

}  // namespace lbann
