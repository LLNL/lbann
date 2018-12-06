////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "conduit/conduit_relay_mpi.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include <time.h>

using namespace lbann;

const int lbann_default_random_seed = 42;

#define NUM_OUTPUT_DIRS 100
#define NUM_SAMPLES_PER_FILE 1000

//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (!opts->has_string("filelist")) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string>");
      }
    }

    std::vector<std::string> files;
    std::string f;
    int size;
    if (master) {
      std::stringstream s;
      std::ifstream in(opts->get_string("filelist").c_str());
      if (!in) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + opts->get_string("filelist") + " for reading");
      }
      std::string line;
      while (getline(in, line)) {
        if (line.size()) {
          s << line << " ";
          //files.push_back(line);
        }
      }
      in.close();
      f = s.str();
      size = s.str().size();
      std::cout << "size: " << size << "\n";
    }
    comm->world_broadcast<int>(0, &size, 1);
    f.resize(size);
    comm->world_broadcast<char>(0, &f[0], size);

    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        files.push_back(filename);
      }
    }

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    size_t h = 0;
    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";

      try {

        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (std::exception e) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n"; 
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (std::exception e) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      }

      for (size_t i=0; i<cnames.size(); i++) {

        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (std::exception e) {
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        }  

        int success = n_ok.to_int64();
        if (success == 1) {
            try {
              key = cnames[i] + "/inputs";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
  
              key = cnames[i] + "/outputs/scalars";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);

              key = cnames[i] + "/outputs/images";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
  
#if 0
              key = cnames[i] + "/outputs/images/(0.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
  
              key = cnames[i] + "/outputs/images/(90.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
  
              key = cnames[i] + "/outputs/images/(90.0, 78.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
#endif
  
            } catch (std::exception e) {
              std::cerr << rank << " :: " << "exception caught during extraction: " << cnames[i] << " " << files[j] << "\n";
              continue;
            }
        }
      }
    }

  } catch (exception const &e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  }

  // Clean up
  finalize(comm);
  return EXIT_SUCCESS;
}
#endif //#ifdef LBANN_HAS_CONDUIT
