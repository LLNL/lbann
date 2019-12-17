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
////////////////////////////////////////////////////////////////////////////////

#include "lbann_config.hpp"
#include "conduit/conduit.hpp"
#include "hdf5.h"
#include <string>
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/options.hpp"

#define _USE_IO_HANDLE_
//#undef _USE_IO_HANDLE_

using namespace lbann;

#ifdef _USE_IO_HANDLE_
  #include "lbann/data_readers/sample_list_conduit_io_handle.hpp"
  using sample_list_t = sample_list_conduit_io_handle<std::string>;
#else
  #include "lbann/data_readers/sample_list_hdf5.hpp"
  using sample_list_t = sample_list_hdf5<std::string>;
#endif


int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (!opts->has_string("lbann")) {
      if (master) {
        std::cout << "usage: " << argv[0] << " --lbann=<lbann home>\n";
      }
      return EXIT_FAILURE;
    }

    // Get the sample list file name
    const std::string base = opts->get_string("lbann");
    #ifdef _USE_IO_HANDLE_
    const std::string sample_list_file = base + "/model_zoo/jag_utils/test_data/test_sample_list_bin.txt";
    #else
    const std::string sample_list_file = base + "/model_zoo/jag_utils/test_data/test_sample_list_hdf5.txt";
    #endif

    // Load the sample list
    sample_list_t sl;
    size_t stride = comm->get_procs_per_trainer();
    size_t offset = comm->get_rank_in_trainer();
    if (master) std::cout << "\nCalling: sample_list.load(..)\n";
    sl.load(sample_list_file, stride, offset);
    if (master) std::cout << "\nCalling: sample_list.all_gather_packed_lists(..)\n";
    sl.all_gather_packed_lists(*comm);
    if (master) std::cout << "\nTests passed!\n\n";

  } catch (std::exception const &e) {
    if (master) std::cerr << "caught exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}

