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
        std::cerr << "usage: " << argv[0] << " --filelist=<string> --output_fn=<string>" << std::endl;
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

    size_t total_samples = 0;
    std::vector<double> v(14, 0.);
    std::vector<double> v_min(14, std::numeric_limits<double>::max());
    size_t n = 0;
    for (size_t j=rank; j<filenames.size(); j+=np) {
      if (master) {
        std::cerr << "loading: " << filenames[j] << std::endl;
      }
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      // get number of samples in the files
      for (const auto &t : a) {
        const std::vector<size_t> &shape = t.second.shape;
        n = shape[0];
        total_samples += n;
        break;
      }

      size_t n_elts = a["density_sig1"].num_vals;
      double *data = reinterpret_cast<double*>(a["density_sig1"].data_holder->data());

      int s = 0;
      for (size_t i=0; i<n_elts; i++) {
        double vv = data[i];
        if (vv > v[s]) v[s] = vv;
        if (vv < v_min[s]) v_min[s] = vv;
        ++s;
        if (s == 14) {
          s = 0;
        }
      }
      if (master) {
        std::cerr << "approx " << utils::commify(total_samples*np) << " samples processed" << std::endl;
      }
    }
    // ==================== finished processing all files ========================

    std::vector<double> f(14, 0.);
    size_t n3 = 0;
    if (rank == 0) {
      n3 = comm->trainer_reduce(total_samples);
      comm->trainer_reduce(v.data(), v.size(), f.data(), El::mpi::MAX);
    } else {
      comm->trainer_reduce(total_samples, 0);
      comm->trainer_reduce(v.data(), v.size(), 0, El::mpi::MAX);
    }

    if (master) {
      std::cerr << "\nactual num samples processed: " << utils::commify(n3) << std::endl;
      std::cerr << "channel normalization values: ";
      for (auto t : f) {
        std::cerr << t << " ";
      }
      std::cerr << std::endl;
      std::ofstream out(output_fn.c_str());
      for (size_t i=0; i<v.size(); i++) {
        out << "      " << v[i] << " " << v_min[i] << "\n";
      }
      /*
       * TODO: perhaps put this in prototext, similar to data_reder_jag_conduit
      out << "data_set_metadata_pilot2 {\n"
          << "  pilot2_normalization {\n"
          << "    channel_normalization_params: [\n";
      for (auto t : f) {
        out << "      " << t << "\n";
      }
      out << "    ]\n"
          << "  }\n"
          << "}\n";
      */
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

