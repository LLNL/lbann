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

#ifndef LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_IO_LAYER_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/data_readers/lbann_data_reader.hpp"
#include "lbann/utils/lbann_dataset.hpp"
#include "lbann/io/lbann_persist.hpp"

// snprintf
#include <stdio.h>

namespace lbann {
class io_layer : public Layer {
 public:
  io_layer(data_layout data_dist, lbann_comm *comm, uint mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers, std::vector<regularizer *> regs= {}, bool data_sets_span_models=true, bool for_regression=false);
  //    io_layer(lbann_comm* comm, uint mini_batch_size, generic_data_reader* training_data_reader);
  void setup_data_readers_for_training(int base_offset, int batch_stride, int sample_stride = 1, int model_offset = 0);
  void setup_data_readers_for_evaluation(int base_offset, int batch_stride, int sample_stride = 1, int model_offset = 0);
  generic_data_reader *select_data_reader();
  generic_data_reader *set_training_data_reader(generic_data_reader *data_reader);
  generic_data_reader *set_validation_data_reader(generic_data_reader *data_reader);
  generic_data_reader *set_testing_data_reader(generic_data_reader *data_reader);
  long update_num_samples_processed(long num_samples);

  long get_num_samples_trained() {
    return m_training_dataset.num_samples_processed;
  }
  long get_num_samples_tested() {
    return m_testing_dataset.num_samples_processed;
  }
  long get_total_num_training_samples() {
    return m_training_dataset.total_samples;
  }
  long get_total_num_testing_samples() {
    return m_testing_dataset.total_samples;
  }

  El::Matrix<El::Int>& get_sample_indices_per_mb();

  bool at_new_epoch() {
    return m_training_dataset.data_reader->at_new_epoch();
  }

  long get_linearized_data_size();
  long get_linearized_label_size();
  long get_linearized_response_size(void) const {
    return static_cast<long>(1);
  }

  // save state of IO to a checkpoint
  bool saveToCheckpointShared(persist& p);
  bool loadFromCheckpointShared(persist& p);

 protected:
  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;
  bool m_data_sets_span_models;

 private:
  const bool m_for_regression;

 public:
  bool is_for_regression(void) const {
    return m_for_regression;
  }
};
}

#endif  // LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
