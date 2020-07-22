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

using namespace lbann;

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

    size_t nn = 0; // only for user feedback
    std::vector<float> max(3, FLT_MIN);
    std::vector<float> min(3, FLT_MAX);
    std::vector<double> total(3, 0.); //for computing mean
    size_t count = 0;            //for compputing mean
    for (size_t j=rank; j<filenames.size(); j+=np) {

      // Get num samples, and sanity check
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      const std::vector<size_t> shape = a["bbs"].shape;
      const size_t num_frames = shape[0];
      const size_t word_size = a["bbs"].word_size;
      bool is_good = true;
      if (shape[1] != 184 || shape[2] != 3 || word_size != 4) {
        is_good = false;
        std::stringstream s3;
        for (auto t : shape) { s3 << t << " "; }
        LBANN_WARNING("Bad file: ", filenames[j], " word_size: ", word_size, " dinum_frames: ", num_frames, " shape: ", s3.str());
      }

      if (is_good) {

        // Get the bbs data array
        const float *data = a["bbs"].data<float>();

        // Loop over the bbs entries
        for (size_t k=0; k<num_frames*184; k++) {
          float xx = data[0];
          float yy = data[1];
          float zz = data[2];
          if (xx < min[0]) min[0] = xx;
          if (xx > max[0]) max[0] = xx;
          if (yy < min[1]) min[1] = yy;
          if (yy > max[1]) max[1] = yy;
          if (zz < min[2]) min[2] = zz;
          if (zz > max[2]) max[2] = zz;
          total[0] += xx;
          total[1] += yy;
          total[2] += zz;
          data += 3;
          ++count;
        }

        ++nn;
        if (!rank) {
          std::cout << "approx " << utils::commify(nn*np) << " files of "
          << utils::commify(filenames.size()) << " processed\n";
        }
      }
    } // END: for (size_t j=rank; j<filenames.size(); j+=np)

    // Collect and report global min/max values
    // (using MPI native calls because having separate calls for root/non-root
    // processes is just annoying. We also have well over a dozen reduce
    // methods, and I can never remember which to use
    std::vector<float> max_all(3);
    std::vector<float> min_all(3);
    std::vector<double> mean(3);
    size_t count_all;

    // only master needs to know min and max
    MPI_Reduce(max.data(), max_all.data(), 3, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(min.data(), min_all.data(), 3, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    // all ranks need to know totals and num_samples, in order to compute
    // std deviation
    MPI_Allreduce(total.data(), mean.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&count, &count_all, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    for (size_t i=0; i<3; i++) {
      mean[i] /= count_all;
    }

    // compute standard deviation
    std::vector<double> v_minus_mean_squared(3, 0);
    for (size_t j=rank; j<filenames.size(); j+=np) {
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      const std::vector<size_t> shape = a["bbs"].shape;
      const size_t word_size = a["bbs"].word_size;
      const size_t num_samples = shape[0];
      bool is_good = true;
      if (shape[1] != 184) { is_good = false; }
      if (shape[2] != 3) { is_good = false; }
      if (word_size != 4) { is_good = false; }
      if (is_good) {
        const float *data = a["bbs"].data<float>();
        for (size_t k=0; k<num_samples*184; k++) {
          float xx = data[0];
          float yy = data[1];
          float zz = data[2];
          v_minus_mean_squared[0] += ((xx - mean[0])*(xx - mean[0]));
          v_minus_mean_squared[1] += ((yy - mean[1])*(yy - mean[1]));
          v_minus_mean_squared[2] += ((zz - mean[2])*(zz - mean[2]));
          data += 3;
        }
      }
    }
    std::vector<double> all_minus_mean_squared(3, 0.);
    std::vector<double> std_dev(3, 0.);
    MPI_Reduce(v_minus_mean_squared.data(), all_minus_mean_squared.data(), 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
      for (size_t i=0; i<3; i++) {
        double v3 = all_minus_mean_squared[i] / count_all;
        std_dev[i] = sqrt(v3);
      }

      std::cout << "\nmax x/y/z: ";
      for (auto t : max_all) std::cout << t << " ";
      std::cout << std::endl;
      std::cout << "min x/y/z: ";
      for (auto t : min_all) std::cout << t << " ";
      std::cout << std::endl;
      std::cout << "mean x/y/z: ";
      for (auto t : mean) std::cout << t << " ";
      std::cout << std::endl;
      std::cout << "std dev: ";
      for (auto t : std_dev) std::cout << t << " ";
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
