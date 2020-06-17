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

#include <vector>
#include "lbann/comm.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/commify.hpp"
#include <cnpy.h>
#include <math.h>

using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = 0;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);

    if (argc == 1) {
      if (master) {
        std::cerr << "usage: " << argv[0] << " --filelist=<string>" << std::endl;
      }
      return EXIT_FAILURE;
    }

    if (! (opts->has_string("filelist"))) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: improper invocation; run with no cmd line args for proper invocation");
    }

    const std::string input_fn = opts->get_string("filelist");

    int rank = comm->get_rank_in_world();
    int np = comm->get_procs_in_world();

    char b[1024];
    sprintf(b, "debug.%d", rank);
    std::ofstream out(b);
    if (! out) {
      LBANN_ERROR("failed to open ", b, " for reading");
    }

    // get list of input filenames
    std::vector<std::string> filenames;
    read_filelist(comm.get(), input_fn, filenames);

    for (size_t j=rank; j<filenames.size(); j+=np) {
      if (master) {
        std::cerr << "loading: " << filenames[j] << std::endl;
      }

      out << "opening: " << filenames[j] << std::endl;
      out.close();
      out.open(b, std::ofstream::out | std::ofstream::app);

      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);

      out << "DONE! opening: " << filenames[j] << std::endl;
      out.close();
      out.open(b, std::ofstream::out | std::ofstream::app);
    }
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

