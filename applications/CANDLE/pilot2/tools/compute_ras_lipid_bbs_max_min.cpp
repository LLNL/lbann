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

//#include <vector>
#include "lbann/comm.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/commify.hpp"
#include <cnpy.h>
#include <cmath>
#include <cfloat>

using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = 0;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (! opts->has_string("filelist")) {
      LBANN_ERROR("usage: ", argv[0], " --filelist=<string>");
    }

    std::string input_fn = opts->get_string("filelist");

    int rank = comm->get_rank_in_world();
    int np = comm->get_procs_in_world();

    // get list of input filenames
    std::vector<std::string> filenames;
    read_filelist(comm.get(), input_fn, filenames);

    size_t nn = 0; // only for user feedback
    std::vector<float> max(3, FLT_MIN);
    std::vector<float> min(3, FLT_MAX);
    for (size_t j=rank; j<filenames.size(); j+=np) {

      // Get num samples, and sanity check
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      const std::vector<size_t> shape = a["bbs"].shape;
      const size_t num_samples = shape[0];
      bool is_good = true;
      if (shape[1] != 184) {
        LBANN_WARNING("shape[1] != 184; shape[1]= ", shape[1], " for file: ", filenames[j], "; shape[1]: ", shape[1], " shape[2]: ", shape[2]);
        is_good = false;
      }
      if (shape[2] != 3) {
        LBANN_WARNING("shape[2] != 3; shape[1]= ", shape[1], " for file: ", filenames[j], "; shape[1]: ", shape[1], " shape[2]: ", shape[2]);
        is_good = false;
      }
      const size_t word_size = a["bbs"].word_size;
      if (word_size != 4) {
        LBANN_WARNING("word_size != 4; word_size: ", word_size, " for file: ", filenames[j]);
        is_good = false;
      }

      if (is_good) {
  
        // Get the bbs data array
        const float *data = a["bbs"].data<float>();
  
        // Loop over the bbs entries
        for (size_t k=0; k<num_samples; k++) {
          float xx = data[0];
          float yy = data[1];
          float zz = data[2];
          if (xx < min[0]) min[0] = xx;
          if (xx > max[0]) max[0] = xx;
          if (yy < min[1]) min[1] = yy;
          if (yy > max[1]) max[1] = yy;
          if (zz < min[2]) min[2] = zz;
          if (zz > max[2]) max[2] = zz;
          data += 3;
        }  
  
        ++nn;
        if (!rank) {
          std::cout << "approx " << (nn*np) << " files of " 
          << filenames.size() << " processed\n";
        }
      }
    } // END: for (size_t j=rank; j<filenames.size(); j+=np) 

    // Collect and report global min/max values
    // (using MPI native calls because having separate calls for root/non-root
    // processes is just annoying. We also have well over a dozen reduce
    // methods, and I can never remember which to use
    std::vector<float> max_all(3);
    std::vector<float> min_all(3);
    MPI_Reduce(max.data(), max_all.data(), 3, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(min.data(), min_all.data(), 3, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (!rank) {
      std::cout << "\nmax x/y/z: ";
      for (auto t : max_all) std::cout << t << " ";
      std::cout << std::endl;
      std::cout << "min x/y/z: ";
      for (auto t : min_all) std::cout << t << " ";
      std::cout << std::endl;
    }

  } catch (std::exception const &e) {
    if (master) std::cerr << "caught exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

