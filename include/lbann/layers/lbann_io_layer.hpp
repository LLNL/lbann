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

// snprintf
#include <stdio.h>

namespace lbann
{
  class io_layer : public Layer {
  public:
    io_layer(lbann_comm* comm, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, std::vector<regularizer*> regs={}, bool data_sets_span_models=true, bool for_regression=false);
    //    io_layer(lbann_comm* comm, uint mini_batch_size, DataReader* training_data_reader);
    void setup_data_readers_for_training(int base_offset, int stride, int model_offset = 0);
    void setup_data_readers_for_evaluation(int base_offset, int stride, int model_offset = 0);
    DataReader *select_data_reader();
    DataReader *set_training_data_reader(DataReader *data_reader);
    DataReader *set_validation_data_reader(DataReader *data_reader);
    DataReader *set_testing_data_reader(DataReader *data_reader);
    long update_num_samples_processed(long num_samples);

    long get_num_samples_trained() { return m_training_dataset.num_samples_processed; }
    long get_num_samples_tested() { return m_testing_dataset.num_samples_processed; }
    long get_total_num_training_samples() { return m_training_dataset.total_samples; }
    long get_total_num_testing_samples() { return m_testing_dataset.total_samples; }

    long get_linearized_data_size();
    long get_linearized_label_size();
    long get_linearized_response_size(void) const { return static_cast<long>(1); }

    struct dataset_header {
        long train_proc;
        long train_total;
        long test_proc;
        long test_total;
        long validate_proc;
        long validate_total;
    };

    // save state of IO to a checkpoint
    bool saveToCheckpointShared(const char* dir, uint64_t* bytes) {
        // get our rank and the number of ranks
        int rank, ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);

        // rank 0 writes the file
        if (rank == 0) {
            // define a filename for this layer
            char filename[1024];
            snprintf(filename, sizeof(filename), "%s/L%d_IO", dir, Index);

            dataset_header header;
            header.train_proc     = m_training_dataset.num_samples_processed;
            header.train_total    = m_training_dataset.total_samples;
            header.test_proc      = m_testing_dataset.num_samples_processed;
            header.test_total     = m_testing_dataset.total_samples;
            header.validate_proc  = m_validation_dataset.num_samples_processed;
            header.validate_total = m_validation_dataset.total_samples;

            // open the file for writing
            int fd = lbann::openwrite(filename);

            // write the header
            ssize_t write_rc = write(fd, &header, sizeof(header));
            if (write_rc != sizeof(header)) {
                // error!
            }
            *bytes += write_rc;

            // close our file
            lbann::closewrite(fd, filename);
        }

        return true;
    }

    bool loadFromCheckpointShared(const char* dir, uint64_t* bytes) {
        // get our rank and the number of ranks
        int rank, ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);

        // rank 0 reads the file
        dataset_header header;
        if (rank == 0) {
            // define a filename for this layer
            char filename[1024];
            snprintf(filename, sizeof(filename), "%s/L%d_IO", dir, Index);

            // open the file for reading
            int fd = lbann::openread(filename);

            // read the header
            ssize_t read_rc = read(fd, &header, sizeof(header));
            if (read_rc != sizeof(header)) {
                // error!
            }
            *bytes += read_rc;

            // close our file
            lbann::closeread(fd, filename);
        }

        // TODO: assumes homogeneous hardware
        // broadcast data from rank 0
        MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

        // set our fields
        m_training_dataset.num_samples_processed   = header.train_proc;
        m_training_dataset.total_samples           = header.train_total;
        m_testing_dataset.num_samples_processed    = header.test_proc;
        m_testing_dataset.total_samples            = header.test_total;
        m_validation_dataset.num_samples_processed = header.validate_proc;
        m_validation_dataset.total_samples         = header.validate_total;

        return true;
    }

  public:
    dataset m_training_dataset;
    dataset m_testing_dataset;
    dataset m_validation_dataset;
    bool m_data_sets_span_models;

  private:
    const bool m_for_regression;
  public:
    bool is_for_regression(void) const { return m_for_regression; }
  };
}

#endif  // LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
