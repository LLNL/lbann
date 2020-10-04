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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LIBRARY_HPP
#define LBANN_LIBRARY_HPP

#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"

namespace lbann {

const int lbann_default_random_seed = 42;

#define MAX_RNG_SEEDS_DISPLAY "RNG seeds per trainer to display"
#define NUM_IO_THREADS "Num. IO threads"
#define NUM_TRAIN_SAMPLES "Num train samples"
#define NUM_VALIDATE_SAMPLES "Num validate samples"
#define NUM_TEST_SAMPLES "Num test samples"
#define ALLOW_GLOBAL_STATISTICS "LTFB Allow global statistics"

void construct_std_options();

std::unique_ptr<trainer> construct_trainer(lbann_comm *comm,
                                           lbann_data::Trainer* pb_trainer,
                                           lbann_data::LbannPB &pb,
                                           options *opts);

std::unique_ptr<thread_pool> construct_io_thread_pool(lbann_comm *comm, options *opts);

std::unique_ptr<model> build_model_from_prototext(
    int argc, char **argv,
    const lbann_data::Trainer* pb_trainer,
    lbann_data::LbannPB &pb,
    lbann_comm *comm,
    options *opts,
    thread_pool& io_thread_pool,
    std::vector<std::shared_ptr<callback_base>>& shared_callbacks,
    int training_dr_linearized_data_size);

void print_lbann_configuration(lbann_comm *comm,
                               int io_threads_per_process,
                               int io_threads_offset);

} // namespace lbann

#endif // LBANN_LIBRARY_HPP
