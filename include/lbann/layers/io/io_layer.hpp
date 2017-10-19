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

#include "lbann/layers/layer.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/dataset.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/exception.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// snprintf
#include <stdio.h>

namespace lbann {
class io_layer : public Layer {
 protected:
  // dataset m_training_dataset;
  // dataset m_testing_dataset;
  // dataset m_validation_dataset;
  bool m_data_sets_span_models;

 private:
  bool m_for_regression;

 public:
  io_layer(lbann_comm *comm,
           //           std::map<execution_mode, generic_data_reader *> data_readers,
           bool data_sets_span_models = true,
           bool for_regression = false)
    : Layer(0, comm),
      // m_training_dataset(data_readers[execution_mode::training]),
      // m_testing_dataset(data_readers[execution_mode::testing]),
      // m_validation_dataset(data_readers[execution_mode::validation]),
      m_data_sets_span_models(data_sets_span_models),
      m_for_regression(for_regression) {
    // if(m_training_dataset.m_data_reader != nullptr) {
    //   m_training_dataset.m_total_samples = m_training_dataset.m_data_reader->get_num_data();
    // }

    // if(m_validation_dataset.m_data_reader != nullptr) {
    //   m_validation_dataset.m_total_samples = m_validation_dataset.m_data_reader->get_num_data();
    // }

    // if(m_testing_dataset.m_data_reader != nullptr) {
    //   m_testing_dataset.m_total_samples = m_testing_dataset.m_data_reader->get_num_data();
    // }
  }

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    Layer::initialize_distributed_matrices<T_layout>();
  }

  virtual void set_paired_input_layer(io_layer *input_layer) {};

  // /**
  //  * Use the data readers in layer l.
  //  */
  // void set_data_readers_from_layer(io_layer *l) {
  //   m_training_dataset = l->m_training_dataset;
  //   m_validation_dataset = l->m_validation_dataset;
  //   m_testing_dataset = l->m_testing_dataset;
  // }

  // generic_data_reader *set_training_data_reader(generic_data_reader *data_reader) {
  //   /// @todo put in a check to make sure that this is a data reader
  //   /// that matches what was already there
  //   generic_data_reader *old_data_reader = m_training_dataset.m_data_reader;
  //   m_training_dataset.m_data_reader = data_reader;
  //   m_training_dataset.m_num_samples_processed = 0;
  //   m_training_dataset.m_total_samples = data_reader->get_num_data();
  //   return old_data_reader;
  // }

  // generic_data_reader *set_validation_data_reader(generic_data_reader *data_reader) {
  //   /// @todo put in a check to make sure that this is a data reader
  //   /// that matches what was already there
  //   generic_data_reader *old_data_reader = m_validation_dataset.m_data_reader;
  //   m_validation_dataset.m_data_reader = data_reader;
  //   m_validation_dataset.m_num_samples_processed = 0;
  //   m_validation_dataset.m_total_samples = data_reader->get_num_data();
  //   return old_data_reader;
  // }

  // generic_data_reader *set_testing_data_reader(generic_data_reader *data_reader) {
  //   /// @todo put in a check to make sure that this is a data reader
  //   /// that matches what was already there
  //   generic_data_reader *old_data_reader = m_testing_dataset.m_data_reader;
  //   m_testing_dataset.m_data_reader = data_reader;
  //   m_testing_dataset.m_num_samples_processed = 0;
  //   m_testing_dataset.m_total_samples = data_reader->get_num_data();
  //   return old_data_reader;
  // }

  /**
   * Return the dataset for the given execution mode.
   */
  virtual dataset& get_dataset(execution_mode m) = 0;

  /**
   * Return the dataset associated with the current execution mode.
   */
  virtual dataset& select_dataset() = 0;

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  virtual dataset* select_first_valid_dataset() = 0;

  /**
   * Return the data reader associated with the current execution mode.
   */
  virtual generic_data_reader *select_data_reader() = 0;

  /**
   * Update the number of samples processed for the current execution mode.
   */
  virtual long update_num_samples_processed(long num_samples) = 0;

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
  virtual El::Matrix<El::Int>* get_sample_indices_per_mb() = 0;

  /**
   * Get the dimensions of the underlying data.
   */
  virtual const std::vector<int> get_data_dims() = 0;

  virtual std::string get_topo_description() const = 0;

  /**
   * Get the linearized size of the underlying data.
   */
  virtual long get_linearized_data_size() = 0;

  /**
   * Get the linearized size of the labels for the underlying data.
   */
  virtual long get_linearized_label_size() = 0;

  virtual long get_linearized_response_size() const = 0;

  virtual long get_num_samples_trained() = 0;
  virtual long get_num_samples_tested() = 0;
  virtual long get_total_num_training_samples() = 0;
  virtual long get_total_num_testing_samples() = 0;

  virtual bool at_new_epoch() = 0;

  virtual bool is_execution_mode_valid(execution_mode mode) = 0;

#if 0
  bool saveToCheckpointShared(persist& p) {
    // rank 0 writes the file
    if (p.get_rank() == 0) {
      p.write_uint64(persist_type::train, "reader_train_processed",
                     (uint64_t) m_training_dataset.m_num_samples_processed);
      p.write_uint64(persist_type::train, "reader_train_total",
                     (uint64_t) m_training_dataset.m_total_samples);

      p.write_uint64(persist_type::train, "reader_test_processed",
                     (uint64_t) m_testing_dataset.m_num_samples_processed);
      p.write_uint64(persist_type::train, "reader_test_total",
                     (uint64_t) m_testing_dataset.m_total_samples);

      p.write_uint64(persist_type::train, "reader_validate_processed",
                     (uint64_t) m_validation_dataset.m_num_samples_processed);
      p.write_uint64(persist_type::train, "reader_validate_total",
                     (uint64_t) m_validation_dataset.m_total_samples);
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
    m_training_dataset.m_num_samples_processed   = (long) header.train_proc;
    m_training_dataset.m_total_samples           = (long) header.train_total;
    m_testing_dataset.m_num_samples_processed    = (long) header.test_proc;
    m_testing_dataset.m_total_samples            = (long) header.test_total;
    m_validation_dataset.m_num_samples_processed = (long) header.validate_proc;
    m_validation_dataset.m_total_samples         = (long) header.validate_total;

    return true;
  }
#endif
  bool is_for_regression() const {
    return m_for_regression;
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
