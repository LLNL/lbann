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
#include <time.h>

using namespace lbann;

#define NUM_OUTPUT_DIRS 100
#define NUM_SAMPLES_PER_FILE 1000

//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();


  try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (!opts->has_string("filelist")) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist");
      }
    }

    std::vector<std::string> files;
    std::ifstream in(opts->get_string("filelist").c_str());
    if (!in) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + opts->get_string("filelist") + " for reading");
    }
    std::string line;
    while (getline(in, line)) {
      if (line.size()) {
        files.push_back(line);
      }
    }
    in.close();

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;

    size_t h = 0;
    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";
      try {
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j] );
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (const std::exception&) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names: " << files[j] << "\n";
        continue;
      }

      for (size_t i=0; i<cnames.size(); i++) {
        // is the next sample valid?
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (const exception& e) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: caught exception reading success flag for child " + std::to_string(i) + " of " + std::to_string(cnames.size()) + "; " + e.what());
        }
        int success = n_ok.to_int64();

        if (success == 1) {
          key = "/" + cnames[i] + "/outputs/images";
          std::vector<std::string> image_names;
          try {
            conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, key, image_names);
          } catch (const std::exception&) {
            std::cerr << rank << " :: exception :hdf5_group_list_child_names for images: " << files[j] << "\n";
            continue;
          }
        }
      }
    }
  } catch (const std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}
