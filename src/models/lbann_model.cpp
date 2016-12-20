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
// lbann_model .hpp .cpp - Abstract class for neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/lbann_model.hpp"
#include "lbann/callbacks/lbann_callback.hpp"
#include <string>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::model::model(lbann_comm* comm, objective_fn* obj_fn) :
  obj_fn(obj_fn),
  m_execution_mode(execution_mode::invalid),
  m_terminate_training(false),
  m_current_epoch(0), m_current_step(0),
  m_current_validation_step(0), m_current_testing_step(0),
  m_current_mini_batch_size(0),m_current_phase(0),
  comm(comm)
{
}

void lbann::model::add_callback(lbann::lbann_callback* cb) {
  callbacks.push_back(cb);
}

void lbann::model::setup_callbacks() {
  for (auto&& cb : callbacks) {
    cb->setup(this);
  }
}

void lbann::model::do_train_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_train_begin(this);
  }
}

void lbann::model::do_train_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_train_end(this);
  }
}

void lbann::model::do_phase_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_phase_end(this);
  }
}

void lbann::model::do_epoch_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_epoch_begin(this);
  }
}

void lbann::model::do_epoch_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_epoch_end(this);
  }
}

void lbann::model::do_batch_begin_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_batch_begin(this);
    }
  }
}

void lbann::model::do_batch_end_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_batch_end(this);
    }
  }
}

void lbann::model::do_test_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_test_begin(this);
  }
}

void lbann::model::do_test_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_test_end(this);
  }
}

void lbann::model::do_validation_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_validation_begin(this);
  }
}

void lbann::model::do_validation_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_validation_end(this);
  }
}

void lbann::model::do_model_forward_prop_begin_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_forward_prop_begin(this);
    }
  }
}

void lbann::model::do_layer_forward_prop_begin_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_forward_prop_begin(this, l);
    }
  }
}

void lbann::model::do_model_forward_prop_end_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_forward_prop_end(this);
    }
  }
}

void lbann::model::do_layer_forward_prop_end_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_forward_prop_end(this, l);
    }
  }
}

void lbann::model::do_model_backward_prop_begin_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_backward_prop_begin(this);
    }
  }
}

void lbann::model::do_layer_backward_prop_begin_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_backward_prop_begin(this, l);
    }
  }
}

void lbann::model::do_model_backward_prop_end_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_backward_prop_end(this);
    }
  }
}

void lbann::model::do_layer_backward_prop_end_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_backward_prop_end(this, l);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Evaluation callbacks
////////////////////////////////////////////////////////////////////////////////

void lbann::model::do_batch_evaluate_begin_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_batch_evaluate_begin(this);
    }
  }
}

void lbann::model::do_batch_evaluate_end_cbs() {
  for (auto&& cb : callbacks) {
    if (get_cur_step() % cb->batch_interval == 0) {
      cb->on_batch_evaluate_end(this);
    }
  }
}

void lbann::model::do_model_evaluate_forward_prop_begin_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_evaluate_forward_prop_begin(this);
  }
}

void lbann::model::do_layer_evaluate_forward_prop_begin_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    cb->on_evaluate_forward_prop_begin(this, l);
  }
}

void lbann::model::do_model_evaluate_forward_prop_end_cbs() {
  for (auto&& cb : callbacks) {
    cb->on_evaluate_forward_prop_end(this);
  }
}

void lbann::model::do_layer_evaluate_forward_prop_end_cbs(Layer* l) {
  for (auto&& cb : callbacks) {
    cb->on_evaluate_forward_prop_end(this, l);
  }
}

/* struct used to serialize mode fields in file and MPI transfer */
struct lbann_model_header {
    uint32_t execution_mode;
    uint32_t terminate_training;
    int64_t current_epoch;
    int64_t current_step;
};

bool lbann::model::save_to_checkpoint_shared(const char* dir, uint64_t* bytes)
{
    // write a single header describing layers and sizes?

    // get our rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // have rank 0 write the network file
    if (rank == 0) {
        // define filename for training state
        char filename[1024];
        sprintf(filename, "%s/model", dir);

        // open the file for writing
        int fd = lbann::openwrite(filename);

        // fill the structure to write to disk
        struct lbann_model_header header;
        header.execution_mode     = (uint32_t) m_execution_mode;
        header.terminate_training = (uint32_t) m_terminate_training;
        header.current_epoch      = (int64_t)  m_current_epoch;
        header.current_step       = (int64_t)  m_current_step;

        // write out our model state
        ssize_t write_rc = write(fd, &header, sizeof(header));
        if (write_rc != sizeof(header)) {
            fprintf(stderr, "ERROR: Failed to write model state to file `%s' (%d: %s) @ %s:%d\n",
                    filename, errno, strerror(errno), __FILE__, __LINE__
            );
            fflush(stderr);
        }
        *bytes += write_rc;

        // close our file
        lbann::closewrite(fd, filename);
    }

    return true;
}

bool lbann::model::load_from_checkpoint_shared(const char* dir, uint64_t* bytes)
{
    // get our rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // have rank 0 read the file
    struct lbann_model_header header;
    if (rank == 0) {
        // define filename for model state
        char filename[1024];
        sprintf(filename, "%s/model", dir);

        // open the file for reading
        int fd = lbann::openread(filename);
        if (fd != -1) {
            // read state from file
            ssize_t read_rc = read(fd, &header, sizeof(header));
            if (read_rc != sizeof(header)) {
                fprintf(stderr, "ERROR: Failed to read model state from file `%s' (%d: %s) @ %s:%d\n",
                        filename, errno, strerror(errno), __FILE__, __LINE__
                );
                fflush(stderr);
            }
            *bytes += read_rc;

            // close our file
            lbann::closeread(fd, filename);
        }
    }

    // TODO: this assumes homogeneous processors
    // broadcast state from rank 0
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    // fill the structure to write to disk
    m_execution_mode     = (execution_mode) header.execution_mode;
    m_terminate_training = (bool)           header.terminate_training;
    m_current_epoch      = (int64_t)        header.current_epoch;
    m_current_step       = (int64_t)        header.current_step;

    return true;
}
