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
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"

using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  if (master) {
    std::cout << "\n\n==============================================================\n"
              << "STARTING " << argv[0] << " with this command line:\n";
    for (int j=0; j<argc; j++) {
      std::cout << argv[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);

    if (argc == 1) {
      if (master) {
        std::cout << "usage: " << argv[0] << " --filelist=<string> --base_dir=<string> --output_fn=<string>\n"
          "where: filelist contains a list of conduit filenames;\n"
          "       base_dir / <name from filelist> should fully specify\n"
          "       a conduit filepath\n"
          "function: constructs an index that lists number of samples\n"
          "          in each file, indices of invalid samples, etc\n";
      }
      return EXIT_SUCCESS;
    }

    if (! (opts->has_string("filelist") && opts->has_string("output_fn") &&  opts->has_string("base_dir") )) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: improper invocation; run with no cmd line args for proper invocation");
    }

    const std::string input_fn = opts->get_string("filelist");
    const std::string output_fn = opts->get_string("output_fn");
    const std::string base_dir = opts->get_string("base_dir");

    int rank = comm->get_rank_in_world();
    std::stringstream ss;
    ss << output_fn << "." << rank;
    std::ofstream out(ss.str());
    std::cerr << rank << " :: opened for writing: " << ss.str() << "\n";
    if (!out.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + output_fn + " for writing");
    }
    if (master) {
      std::cerr << "writing index file: " << output_fn << "\n";
    }

    // get list of input filenames
    std::vector<std::string> filenames;
    read_filelist(comm.get(), input_fn, filenames);

    int num_samples = 0;
    int num_samples_bad = 0;
    int np = comm->get_procs_in_world();
    hid_t hdf5_file_hnd;
    for (size_t j=rank; j<filenames.size(); j+=np) {
if (j >= 400) break;
      int local_num_samples = 0;
      int local_num_samples_bad = 0;
      std::cerr << rank << " :: processing: " << filenames[j] << "\n";
      try {
        std::string fn = filenames[j];
        size_t start_pos = fn.find(base_dir);
        if (start_pos != std::string::npos) {
          fn.replace(start_pos, base_dir.size(), "");
        }
        const std::string sss = base_dir + '/' + fn;
        out << fn << " ";
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( sss );
      } catch (...) {
         std::cerr << "exception hdf5_open_file_for_reading: " << filenames[j] << "\n";
         continue;
      }
      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (...) {
         std::cerr << "exception hdf5_group_list_child_names\n";
         continue;
      }
      std::stringstream s5;
      conduit::Node n_ok;
      for (size_t h=0; h<cnames.size(); h++) {
        const std::string key_1 = "/" + cnames[h] + "/performance/success";

        // adding this since hydra has one top-level child in each file
        // that is not the root or a complete sample. Instead it's some
        // sort of meta-data 
        bool good = conduit::relay::io::hdf5_has_path(hdf5_file_hnd, key_1);
        if (!good) {
          std::cerr << "missing path: " << key_1 << " (this is probably OK for hydra)\n";
          s5 << cnames[h] << " ";
          ++num_samples_bad;
          ++local_num_samples_bad;
          continue;
        }

        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key_1, n_ok);
        } catch (...) {
           std::cerr << "exception hdf5_read file: " << filenames[j] << "; key: " << key_1 << "\n";
           continue;
        }
        int success = n_ok.to_int64();
        if (success == 1) {
          ++num_samples;
          ++local_num_samples;
        } else {
          s5 << cnames[h] << " ";
          ++num_samples_bad;
          ++local_num_samples_bad;
        }
      }
      out << local_num_samples << " " << local_num_samples_bad << " " << s5.str() << "\n";
      try {
        conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
      } catch (...) {
         std::cerr << "exception hdf5_close_file\n";
         continue;
      }
    }
    out.close();
    comm->global_barrier();

    int global_num_samples;
    int global_num_samples_bad;
    MPI_Reduce(&num_samples, &global_num_samples, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&num_samples_bad, &global_num_samples_bad, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (master) {

      std::ofstream out2("num_samples_tmp");
      if (!out2) {
        LBANN_ERROR("failed to open output file");
      }
      out2 << "CONDUIT_HDF5_EXCLUSION\n" << global_num_samples << " " << global_num_samples_bad
           << " " << filenames.size() << "\n" << base_dir << "\n";
      out2.close();

      std::stringstream s3;
      s3 << "cat num_samples_tmp ";
      for (int k=0; k<np; k++) {
        s3 << output_fn << "." << k << " ";
      }
      s3 << "> " << output_fn;
      system(s3.str().c_str());

      s3.clear();
      s3.str("");
      s3 << "chmod 660 " << output_fn;
      system(s3.str().c_str());
      s3.clear();
      s3.str("");
      s3 << "chgrp brain " << output_fn;
      system(s3.str().c_str());

      s3.clear();
      s3.str("");
      s3 << "rm -f num_samples_tmp ";
      for (int k=0; k<np; k++) {
        s3 << output_fn << "." << k << " ";
      }
      system(s3.str().c_str());
    } // if (master)

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

