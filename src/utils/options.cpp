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

#include "lbann/utils/argument_parser.hpp"

namespace lbann {

void construct_std_options() {
  auto& arg_parser = global_argument_parser();
  arg_parser.add_option(MAX_RNG_SEEDS_DISPLAY,
                        {"--rng_seeds_per_trainer_to_display"},
                        utils::ENV("LBANN_RNG_SEEDS_PER_TRAINER_TO_DISPLAY"),
                        "Limit how many random seeds LBANN should display "
                        "from each trainer",
                        2);
  arg_parser.add_option(NUM_IO_THREADS,
                        {"--num_io_threads"},
                        utils::ENV("LBANN_NUM_IO_THREADS"),
                        "Number of threads available to both I/O and "
                        "initial data transformations for each rank.",
                        64);
  arg_parser.add_option(NUM_TRAIN_SAMPLES,
                        {"--num_train_samples"},
                        utils::ENV("LBANN_NUM_TRAIN_SAMPLES"),
                        "Set the number of training samples to ingest.",
                        0);
  arg_parser.add_option(NUM_VALIDATE_SAMPLES,
                        {"--num_validate_samples"},
                        utils::ENV("LBANN_NUM_VALIDATE_SAMPLES"),
                        "Set the number of validate samples to ingest.",
                        0);
  arg_parser.add_option(NUM_TEST_SAMPLES,
                        {"--num_test_samples"},
                        utils::ENV("LBANN_NUM_TEST_SAMPLES"),
                        "Set the number of testing samples to ingest.",
                        0);
  arg_parser.add_flag(ALLOW_GLOBAL_STATISTICS,
                      {"--ltfb_allow_global_statistics"},
                      utils::ENV("LBANN_LTFB_ALLOW_GLOBAL_STATISTICS"),
                      "Allow the print_statistics callback to report "
                      "global (inter-trainer) summary statistics.");
  arg_parser.add_option(PROCS_PER_TRAINER,
                        {"--procs_per_trainer"},
                        utils::ENV("LBANN_PROCS_PER_TRAINER"),
                        "Number of MPI ranks per LBANN trainer, "
                        "If the field is not set (or set to 0) then "
                        " all MPI ranks are assigned to one trainer."
                        " The number of processes per trainer must "
                        " evenly divide the total number of MPI ranks. "
                        " The number of resulting trainers is "
                        " num_procs / procs_per_trainer.",
                        0);
  arg_parser.add_option(TRAINER_GRID_HEIGHT,
                        {"--trainer_grid_height"},
                        utils::ENV("LBANN_TRAINER_GRID_HEIGHT"),
                        "Height of 2D process grid for each trainer. "
                        "Default grid is approximately square.",
                        -1);
  arg_parser.add_option(SMILES_BUFFER_SIZE,
                        {"--smiles_buffer_size"},
                        utils::ENV("LBANN_SMILES_BUFFER_SIZE"),
                        "Size of the read buffer for the SMILES "
                        "data reader.",
                        16*1024*1024UL);
  arg_parser.add_option("mini_batch_size",
                        {"--mini_batch_size"},
                        "Size of mini batches",
                        0);
}

void construct_vision_options() {
}

} // namespace lbann
