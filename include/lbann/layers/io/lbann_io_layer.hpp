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
#include "lbann/utils/lbann_exception.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// snprintf
#include <stdio.h>

namespace lbann {
class io_layer : public Layer {
 protected:
  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;
  bool m_data_sets_span_models;

 private:
  const bool m_for_regression;

 public:
  io_layer(lbann_comm *comm,
           int mini_batch_size,
           std::map<execution_mode, generic_data_reader *> data_readers,
           bool data_sets_span_models = true,
           bool for_regression = false)
    : Layer(0, comm, mini_batch_size),
      m_training_dataset(data_readers[execution_mode::training]),
      m_testing_dataset(data_readers[execution_mode::testing]),
      m_validation_dataset(data_readers[execution_mode::validation]),
      m_data_sets_span_models(data_sets_span_models),
    m_for_regression(for_regression) {
    if(m_training_dataset.data_reader != NULL) {
      m_training_dataset.total_samples = m_training_dataset.data_reader->getNumData();
    }

    if(m_validation_dataset.data_reader != NULL) {
      m_validation_dataset.total_samples = m_validation_dataset.data_reader->getNumData();
    }

    if(m_testing_dataset.data_reader != NULL) {
      m_testing_dataset.total_samples = m_testing_dataset.data_reader->getNumData();
    }
  }

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    Layer::initialize_distributed_matrices<T_layout>();
  }

  // io_layer(lbann_comm* comm, int mini_batch_size, generic_data_reader *training_data_reader)
  //   : io_layer(comm, mini_batch_size, training_data_reader, NULL, {}) {}

  lbann::generic_data_reader *set_training_data_reader(generic_data_reader *data_reader) {
    /// @todo put in a check to make sure that this is a data reader
    /// that matches what was already there
    generic_data_reader *old_data_reader = m_training_dataset.data_reader;
    m_training_dataset.data_reader = data_reader;
    m_training_dataset.num_samples_processed = 0;
    m_training_dataset.total_samples = data_reader->getNumData();
    m_training_dataset.num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
    return old_data_reader;
  }

  lbann::generic_data_reader *set_validation_data_reader(generic_data_reader *data_reader) {
    /// @todo put in a check to make sure that this is a data reader
    /// that matches what was already there
    generic_data_reader *old_data_reader = m_validation_dataset.data_reader;
    m_validation_dataset.data_reader = data_reader;
    m_validation_dataset.num_samples_processed = 0;
    m_validation_dataset.total_samples = data_reader->getNumData();
    m_validation_dataset.num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
    return old_data_reader;
  }

  lbann::generic_data_reader *set_testing_data_reader(generic_data_reader *data_reader) {
    /// @todo put in a check to make sure that this is a data reader
    /// that matches what was already there
    generic_data_reader *old_data_reader = m_testing_dataset.data_reader;
    m_testing_dataset.data_reader = data_reader;
    m_testing_dataset.num_samples_processed = 0;
    m_testing_dataset.total_samples = data_reader->getNumData();
    m_testing_dataset.num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
    return old_data_reader;
  }

  lbann::generic_data_reader *select_data_reader() {
    switch(m_execution_mode) {
    case execution_mode::training:
      return m_training_dataset.data_reader;
      break;
    case execution_mode::validation:
      return m_validation_dataset.data_reader;
      break;
    case execution_mode::testing:
      return m_testing_dataset.data_reader;
      break;
      // case prediction:
      //   return m_prediction_data_reader;
      //   break;
    default:
      throw -1;
    }
  }

  long update_num_samples_processed(long num_samples) {
    switch(m_execution_mode) {
    case execution_mode::training:
      m_training_dataset.num_samples_processed += num_samples;
      return m_training_dataset.num_samples_processed;
      break;
    case execution_mode::validation:
      m_validation_dataset.num_samples_processed += num_samples;
      return m_validation_dataset.num_samples_processed;
      break;
    case execution_mode::testing:
      m_testing_dataset.num_samples_processed += num_samples;
      return m_testing_dataset.num_samples_processed;
      break;
      // case prediction:
      //   return m_prediction_data_reader;
      //   break;
    default:
      throw lbann_exception("lbann_io_layer: invalid execution phase");
    }
  }

  El::Matrix<El::Int>* get_sample_indices_per_mb() {
    switch(m_execution_mode) {
    case execution_mode::training:
      return &(m_training_dataset.data_reader->m_indices_fetched_per_mb);
      break;
    case execution_mode::validation:
      return &(m_validation_dataset.data_reader->m_indices_fetched_per_mb);
      break;
    case execution_mode::testing:
      return &(m_testing_dataset.data_reader->m_indices_fetched_per_mb);
      break;
    default:
      throw lbann_exception("lbann_io_layer: invalid execution phase");
    }
  }

  long get_linearized_data_size() {
    long linearized_data_size = -1;

    if(m_training_dataset.data_reader != NULL) {
      long tmp_linearized_data_size = m_training_dataset.data_reader->get_linearized_data_size();
      if(linearized_data_size != -1 && linearized_data_size != tmp_linearized_data_size) {
        throw lbann_exception("lbann_io_layer: training data set size does not match the currently established data set size");
      }
      linearized_data_size = tmp_linearized_data_size;
    }

    if(m_validation_dataset.data_reader != NULL) {
      long tmp_linearized_data_size = m_validation_dataset.data_reader->get_linearized_data_size();
      if(linearized_data_size != -1 && linearized_data_size != tmp_linearized_data_size) {
        throw lbann_exception("lbann_io_layer: validation data set size does not match the currently established data set size");
      }
      linearized_data_size = tmp_linearized_data_size;
    }

    if(m_testing_dataset.data_reader != NULL) {
      long tmp_linearized_data_size = m_testing_dataset.data_reader->get_linearized_data_size();
      if(linearized_data_size != -1 && linearized_data_size != tmp_linearized_data_size) {
        throw lbann_exception("lbann_io_layer: testing data set size does not match the currently established data set size");
      }
      linearized_data_size = tmp_linearized_data_size;
    }

    return linearized_data_size;
  }

  long get_linearized_label_size() {
    if (is_for_regression()) {
      return static_cast<long>(1);
    }

    long linearized_label_size = -1;

    if(m_training_dataset.data_reader != NULL) {
      long tmp_linearized_label_size = m_training_dataset.data_reader->get_linearized_label_size();
      if(linearized_label_size != -1 && linearized_label_size != tmp_linearized_label_size) {
        throw lbann_exception("lbann_io_layer: training label set size does not match the currently established label set size");
      }
      linearized_label_size = tmp_linearized_label_size;
    }

    if(m_validation_dataset.data_reader != NULL) {
      long tmp_linearized_label_size = m_validation_dataset.data_reader->get_linearized_label_size();
      if(linearized_label_size != -1 && linearized_label_size != tmp_linearized_label_size) {
        throw lbann_exception("lbann_io_layer: validation label set size does not match the currently established label set size");
      }
      linearized_label_size = tmp_linearized_label_size;
    }

    if(m_testing_dataset.data_reader != NULL) {
      long tmp_linearized_label_size = m_testing_dataset.data_reader->get_linearized_label_size();
      if(linearized_label_size != -1 && linearized_label_size != tmp_linearized_label_size) {
        throw lbann_exception("lbann_io_layer: testing label set size does not match the currently established label set size");
      }
      linearized_label_size = tmp_linearized_label_size;
    }

    return linearized_label_size;
  }

  long get_linearized_response_size(void) const {
    return static_cast<long>(1);
  }


  long get_num_samples_trained(void) {
    return m_training_dataset.num_samples_processed;
  }
  long get_num_samples_tested(void) {
    return m_testing_dataset.num_samples_processed;
  }
  long get_total_num_training_samples(void) {
    return m_training_dataset.total_samples;
  }
  long get_total_num_testing_samples(void) {
    return m_testing_dataset.total_samples;
  }

  bool at_new_epoch(void) {
    return m_training_dataset.data_reader->at_new_epoch();
  }

  void setup_data_readers_for_training(int base_offset, int batch_stride, int sample_stride = 1, int model_offset = 0) {
    if(m_training_dataset.data_reader != NULL) {
      m_training_dataset.data_reader->setup(base_offset, batch_stride, sample_stride, model_offset, m_comm);
    }
  }

  /** Do not spread data readers that are used for evaluation across multiple models.
   * Allow each model instance to use the full data set for evaluation so that each model is fairly compared.
   */
  void setup_data_readers_for_evaluation(int base_offset, int batch_stride, int sample_stride = 1, int model_offset = 0) {
    if(m_validation_dataset.data_reader != NULL) {
      m_validation_dataset.data_reader->setup(base_offset, batch_stride, sample_stride, model_offset, NULL/*m_comm*/);
    }

    if(m_testing_dataset.data_reader != NULL) {
      m_testing_dataset.data_reader->setup(base_offset, batch_stride, sample_stride, model_offset, NULL/*m_comm*/);
    }
    return;
  }

  bool saveToCheckpointShared(persist& p) {
    // rank 0 writes the file
    if (p.get_rank() == 0) {
      p.write_uint64(persist_type::train, "reader_train_processed",
                     (uint64_t) m_training_dataset.num_samples_processed);
      p.write_uint64(persist_type::train, "reader_train_total",
                     (uint64_t) m_training_dataset.total_samples);

      p.write_uint64(persist_type::train, "reader_test_processed",
                     (uint64_t) m_testing_dataset.num_samples_processed);
      p.write_uint64(persist_type::train, "reader_test_total",
                     (uint64_t) m_testing_dataset.total_samples);

      p.write_uint64(persist_type::train, "reader_validate_processed",
                     (uint64_t) m_validation_dataset.num_samples_processed);
      p.write_uint64(persist_type::train, "reader_validate_total",
                     (uint64_t) m_validation_dataset.total_samples);
    }

    return true;
  }

  struct dataset_header {
    uint64_t train_proc;
    uint64_t train_total;
    uint64_t test_proc;
    uint64_t test_total;
    uint64_t validate_proc;
    uint64_t validate_total;
  };

  bool loadFromCheckpointShared(persist& p) {
    // rank 0 reads the file
    dataset_header header;
    if (p.get_rank() == 0) {
      p.read_uint64(persist_type::train, "reader_train_processed",    &header.train_proc);
      p.read_uint64(persist_type::train, "reader_train_total",        &header.train_total);
      p.read_uint64(persist_type::train, "reader_test_processed",     &header.test_proc);
      p.read_uint64(persist_type::train, "reader_test_total",         &header.test_total);
      p.read_uint64(persist_type::train, "reader_validate_processed", &header.validate_proc);
      p.read_uint64(persist_type::train, "reader_validate_total",     &header.validate_total);
    }

    // TODO: assumes homogeneous hardware
    // broadcast data from rank 0
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    // set our fields
    m_training_dataset.num_samples_processed   = (long) header.train_proc;
    m_training_dataset.total_samples           = (long) header.train_total;
    m_testing_dataset.num_samples_processed    = (long) header.test_proc;
    m_testing_dataset.total_samples            = (long) header.test_total;
    m_validation_dataset.num_samples_processed = (long) header.validate_proc;
    m_validation_dataset.total_samples         = (long) header.validate_total;

    return true;
  }

  bool is_for_regression(void) const {
    return m_for_regression;
  }
};
}

#endif  // LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
