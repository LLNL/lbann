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

#include "lbann/utils/std_options.hpp"
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
  arg_parser.add_flag(DATA_STORE_FAIL_ON_MISSING_SAMPLES,
                      {"--ds_fail_on_missing_samples"},
                      utils::ENV("LBANN_DS_FAIL_ON_MISSING_SAMPLES"),
                      "Force the data store to fail on a missing sample "
                      "rather than substituting it with a random sample.");
}

} // namespace lbann
