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

#include "lbann/comm.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/commify.hpp"
#include <cnpy.h>
#include <cmath>
#include <cfloat>
#include "common.hpp"

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

    size_t nn = 0; // only used for user feedback
    std::vector<xyz> beads(Num_beads);  
    for (size_t j=rank; j<filenames.size(); j+=np) {
      if (!rank && j == 0) {
        std::cerr << "Opening for processing: " << filenames[j] << std::endl;
      }

      // Get num samples, and run sanity checks
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      bool is_good =  sanity_check_npz_file(a, filenames[j]);

      // Open output file
      std::string fn = filenames[j] + ".bbs_stats";
      if (!is_good) {
        fn += ".bad";
      }
      std::ofstream out(fn.c_str(), std::ios::binary);
      if (!out) {
        LBANN_ERROR("failed to open ", fn, "for writing");
      }

      if (is_good) {
        const std::vector<size_t> shape = a["bbs"].shape;
        const float num_frames = static_cast<float>(shape[0]);

        // output number of frames and beads
        out.write((char*)&num_frames, sizeof(float));
        float nbeads = static_cast<float>(Num_beads);
        out.write((char*)&nbeads, sizeof(float));
  
        // Get the bbs data array
        const float *bd = a["bbs"].data<float>();

        // Loop over the samples (frames)
        for (int k=0; k<num_frames; k++) {
          // Cache all RAS BB bead coordinates for the current sample
          beads.clear();
          for (size_t i=0; i<Num_beads; i++) {
            beads.push_back(xyz(bd[0], bd[1], bd[2]));
            bd += 3;
          }
          
          // Write output for the current sample
          //
          // z-coordinates
          for (size_t g=0; g<Num_beads; g++) {
            out.write((char*)&beads[g].z, sizeof(float));
          }  

          // euclidean distance between each pair of beads i, h,
          // where h >= i
          for (int i=0; i<Num_beads-1; i++) {
            for (int h=i+1; h<Num_beads; h++) {
              float d = beads[i].dist(beads[h]);
              out.write((char*)&d, sizeof(float));
            }
          }
        }
        
        // User feedback
        ++nn;
        if (!rank) {
          std::cerr << "approx " << (nn*np) << " files of " 
          << filenames.size() << " processed\n";
        }

      } // if (is_good)

      // Close output file
      out.close();
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

