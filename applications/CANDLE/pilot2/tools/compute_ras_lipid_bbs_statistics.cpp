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

const int Num_beads = 184;

struct xyz {
  xyz(float xx, float yy, float zz) : x(xx), y(yy), z(zz) { }

  float x;
  float y;
  float z;

  float dist(const xyz &p) {
    return sqrt( 
             (pow( (x-p.x), 2) 
             + pow( (x-p.x), 2) 
             + pow( (x-p.x), 2))
           );  
  }
};

int main(int argc, char *argv[]) {
  int random_seed = 0;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
#if 0
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
    std::vector<xyz> data(Num_beads);  
    for (size_t j=rank; j<filenames.size(); j+=np) {

      // Get num samples, and run sanity checks
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

      // Open output file
      std::string fn = filename[j] + ".bbs_stats";
      if (!is_good) {
        fn += ".bad";
      }
      std::ofstream out(fn.c_str());
      if (!out) {
        LBANN_ERROR("failed to open ", fn, "for writing");
      }

      if (is_good) {
  
        // Get the bbs data array
        const float *data = a["bbs"].data<float>();

        // Loop over the samples (frames) in this file
        for (size_t k=0; k<num_samples; k++) {
  
          // Cache all RAS BB beads coordinates for the current sample
          for (size_t k=0; k<num_samples; k++) {
            data.push_back(xyz(data[0], data[1], data[2]);
            data += 3;
          }  
        }  
  
        ++nn;
        if (!rank) {
          std::cout << "approx " << (nn*np) << " files of " 
          << filenames.size() << " processed\n";
        }
      }

      // Close output file
      out.close();

    } // END: for (size_t j=rank; j<filenames.size(); j+=np) 

#endif
  } catch (std::exception const &e) {
    if (master) std::cerr << "caught exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

