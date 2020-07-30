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

void read_file(const std::string &filename, std::vector<float> &data);

int main(int argc, char *argv[]) {
  world_comm_ptr comm = initialize(argc, argv);
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

    std::vector<float> data;
    std::vector<float> z_coords(Num_beads);
    std::vector<float> distances(Num_dist);

    double max = FLT_MIN;
    double min = FLT_MAX;
    double total= 0;              //for computing mean
    double n_samples = 0;  //for computing mean

    size_t nn = 0;
    for (size_t j=rank; j<filenames.size(); j+=np) {
      const std::string fn = filenames[j] + ".bbs_stats";
      read_file(fn, data);
      const float *w = data.data();
      int n_frames = static_cast<int>(*w++);
      int n_beads = static_cast<int>(*w++);
      if (n_beads != Num_beads) {
        LBANN_ERROR("n_beads != Num_beads; n_beads: ", n_beads, " Num_beads: ", Num_beads);
      }
      for (int h=0; h<n_frames; h++) {
        read_sample(h, data, z_coords, distances);
        int offset = 0;
        for (int i=0; i<Num_beads-1; i++) {
          for (int k=i+1; k<Num_beads; k++) {
            float dist_h_to_i = distances[offset];
            if (dist_h_to_i < min) { min = dist_h_to_i; }
            if (dist_h_to_i > max) { max = dist_h_to_i; }
            total += dist_h_to_i;
            ++n_samples;
            offset++;
          }
        }
      }

      // User feedback
      ++nn;
      if (!rank) {
        std::cerr << "approx " << (nn*np) << " files of "
                  << filenames.size() << " processed\n";
      }
    }

    //==================================================================

    // Collect and report global min/max/mean/std-dev values
    // (using MPI native calls because having separate calls for root/non-root
    // processes is just annoying. We also have well over a dozen reduce
    // methods, and I can never remember which to use)
    //
    double max_all;
    double min_all;
    double total_all;
    double n_samples_all;

    // only master needs to know min and max
    MPI_Reduce(&max, &max_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min, &min_all, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    // all ranks need to know totals and num_samples, in order to compute
    // std deviation
    MPI_Allreduce(&total, &total_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n_samples, &n_samples_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mean = (total_all / n_samples_all);

    // compute standard deviation
    double v_minus_mean_squared = 0.;
    nn = 0;
    for (size_t j=rank; j<filenames.size(); j+=np) {
      const std::string fn = filenames[j] + ".bbs_stats";
      read_file(fn, data);
      const float *w = data.data();
      int n_frames = static_cast<int>(*w++);
      int n_beads = static_cast<int>(*w++);
      if (n_beads != Num_beads) {
        LBANN_ERROR("n_beads != Num_beads");
      }

      for (int h=0; h<n_frames; h++) {
        w += Num_beads; // skip over z-coordinates
        for (int x=0; x<Num_beads-1; x++) {
          for (int y=x+1; y<Num_beads; y++) {
            float v = *w++;
            v_minus_mean_squared += (v-mean)*(v-mean);
          }
        }
      }

      // User feedback
      ++nn;
      if (!rank) {
        std::cerr << "(2) approx " << (nn*np) << " files of "
                  << filenames.size() << " processed\n";
      }
    }

    double all_minus_mean_squared;
    MPI_Reduce(&v_minus_mean_squared, &all_minus_mean_squared, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
      double std_dev;
      double v3 = all_minus_mean_squared / n_samples_all;
      std_dev = sqrt(v3);

      std::cout << "\nmax: " << max_all << std::endl;
      std::cout << "min: " << min_all << std::endl;
      std::cout << "mean: " << mean << std::endl;
      std::cout << "std dev: " << std_dev << std::endl;
      std::cout << "n_samples_all: " << n_samples_all << std::endl;
      std::cout << "n_samples: " << n_samples << std::endl;
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

void read_file(const std::string &filename, std::vector<float> &data) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    LBANN_ERROR("failed to open ", filename, " for reading");
  }
  in.seekg(0, in.end);
  size_t n = in.tellg();
  in.seekg(0, in.beg);
  data.resize(n);
  char *work = reinterpret_cast<char*>(data.data());
  in.read(work, n);
  if (static_cast<size_t>(in.gcount()) != n) {
    LBANN_ERROR("in.gcount() != n (gcount: ", in.gcount(), "; n: ", n, ") for file: ", filename);
  }
}
