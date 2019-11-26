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
        std::cout << "usage: " << argv[0] << " --filelist=<string> --output_fn=<string>" << std::endl;
      }
      return EXIT_FAILURE;
    }

    if (! (opts->has_string("filelist") && opts->has_string("output_fn"))) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: improper invocation; run with no cmd line args for proper invocation");
    }

    const std::string input_fn = opts->get_string("filelist");
    const std::string output_fn = opts->get_string("output_fn");

    //sanity check that we can write to the output file
    if (master) {
      std::ofstream out(output_fn.c_str());
      if (!out) {
        LBANN_ERROR("failed to open ", output_fn, " for writing");
      }
      out.close();
    }

    int rank = comm->get_rank_in_world();
    int np = comm->get_procs_in_world();

    // get list of input filenames
    std::vector<std::string> filenames;
    read_filelist(comm.get(), input_fn, filenames);

    size_t approx = 0;
    size_t exact = 0;
    std::vector<double> v(14, 0.);
    double num_samples = 0;
    for (size_t j=rank; j<filenames.size(); j+=np) {
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      for (const auto &t : a) {
        const std::vector<size_t> &shape = t.second.shape;
        num_samples = shape[0];
        approx += num_samples;
      }  

      size_t n = static_cast<size_t>(num_samples);
      size_t offset = 0;
      for (double r=0; r<n; r++) {
        double *data = reinterpret_cast<double*>(a["density_sig1"].data_holder->data());
        offset += 13*13*14;
        double (&b)[13][13][14] = *reinterpret_cast<double (*)[13][13][14]>(data);
        double max = 0.;
        for (int channel = 0; channel < 14; ++channel) {
          for (size_t k=0; k<13; k++) {
            for (size_t i=0; i<13; i++) {
              double m = b[k][i][channel];
              max = m > max ? m : max;
            }
          }
          v[channel] = max;
        }
        data += offset;
        ++exact;
      }
      if (master) {
        std::cout << "approx " << utils::commify(approx*np) << " samples processed" << std::endl;
      }
    }

    std::vector<double> f(14, 0.);
    if (rank == 0) {
      comm->trainer_reduce(v.data(), v.size(), f.data(), El::mpi::MAX);
    } else {
      comm->trainer_reduce(exact, 0);
      comm->trainer_reduce(v.data(), v.size(), 0, El::mpi::MAX);
    }

    if (master) {
      for (auto t : f) {
        std::cout << t << " ";
      }
      std::cout << std::endl;
      std::ofstream out(output_fn.c_str());
      out << "data_set_metadata_pilot2 {\n"
          << "  pilot2_normalization {\n"
          << "    channel_normalization_params: [\n";
      for (auto t : f) {
        out << "      " << t << "\n";
      }
      out << "    ]\n"
          << "  }\n"
          << "}\n";
      out.close();
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

