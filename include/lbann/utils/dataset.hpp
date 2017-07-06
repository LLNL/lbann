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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATASET_HPP_INCLUDED
#define LBANN_DATASET_HPP_INCLUDED

#include "lbann/data_readers/data_reader.hpp"

namespace lbann {

class dataset {
 public:
  dataset(generic_data_reader *d_reader) : data_reader(d_reader), num_samples_processed(0), total_samples(0) {};
  dataset(const dataset& other) :
    num_samples_processed(other.num_samples_processed),
    total_samples(other.total_samples),
    num_iterations_per_epoch(other.num_iterations_per_epoch) {
    if (data_reader) {
      delete data_reader;
    }
    data_reader = other.data_reader->copy();
  }
  dataset& operator=(const dataset& other) {
    num_samples_processed = other.num_samples_processed;
    total_samples = other.total_samples;
    num_iterations_per_epoch = other.num_iterations_per_epoch;
    if (data_reader) {
      delete data_reader;
    }
    data_reader = other.data_reader->copy();
    return *this;
  }

 public:
  generic_data_reader *data_reader;
  long num_samples_processed;
  long total_samples;
  long num_iterations_per_epoch;
};

}  // namespace lbann

#endif  // LBANN_DATASET_HPP_INCLUDED
