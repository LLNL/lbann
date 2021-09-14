////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

int main(int argc, char *argv[]) {
  world_comm_ptr comm = initialize(argc, argv);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    auto& arg_parser = global_argument_parser();
    construct_std_options();
    construct_jag_options();
    try {
      arg_parser.parse(argc, argv);
    }
    catch (std::exception const& e) {
      auto guessed_rank = guess_global_rank();
      if (guessed_rank <= 0)
        // Cannot call `El::ReportException` because MPI hasn't been
        // initialized yet.
        std::cerr << "Error during argument parsing:\n\ne.what():\n\n  "
                  << e.what() << "\n\nProcess terminating." << std::endl;
      std::terminate();
    }

    // sanity check invocation
    if (arg_parser.get<std::string>(FILELIST) == "" ||
        arg_parser.get<std::string>(OUTPUT_BASE_DIR) == "" ||
        arg_parser.get<int>(NUM_SUBDIRS) == -1 ||
        arg_parser.get<int>(SAMPLES_PER_FILE) == -1) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string> --output_base_dir=<string> --num_subdirs=<int> --samples_per_file=<int>");
      }
    }

    const int num_dirs = arg_parser.get<int>(NUM_SUBDIRS);
    const std::string base = arg_parser.get<std::string>(OUTPUT_BASE_DIR);
    const int samples_per_file = arg_parser.get<int>(SAMPLES_PER_FILE);

    // master creates output directory structure
    if (master) {
      std::stringstream s;
      for (int j=0; j<num_dirs; j++) {
        s << "mkdir -p " << base << "/" << j;
        std::cerr << "\nrunning system call: " << s.str() << "\n\n";
        int r = system(s.str().c_str());
        if (r != 0) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: call to system failed: " + s.str());
        }
        s.clear();
        s.str("");
      }
    }

    // master reads the filelist and bcasts to others
    std::vector<std::string> files;
    std::string f;
    int size;
    if (master) {
      std::stringstream s;
      std::ifstream in(arg_parser.get<std::string>(FILELIST).c_str());
      if (!in) {
        throw lbann_exception(std::string{} + __FILE__ + " " +
                              std::to_string(__LINE__) + " :: failed to open " +
                              arg_parser.get<std::string>(FILELIST) +
                              " for reading");
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

    // unpack the filenames into a vector
    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        files.push_back(filename);
      }
    }
    if (rank == 1) std::cerr << "num files: " << files.size() << "\n";

    // repackage
    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node node;
    conduit::Node save_me;
    size_t h = 0;
    int sample_count = 0;

    int cur_dir = 0;
    int cur_file = 0;
    char output_fn[1024];
    sprintf(output_fn, "%s/%d/samples_%d_%d.bundle", base.c_str(), cur_dir++, cur_file++, rank);

    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";

      try {

        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (const std::exception&) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (const std::exception&) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      }
      std::cerr << rank << " :: " << files[j] << " contains " << cnames.size() << " samples\n";

      for (size_t i=0; i<cnames.size(); i++) {

        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (const std::exception&) {
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
            try {
                key = "/" + cnames[i];
                conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
                save_me["/" + cnames[i]] = node;

            } catch (const std::exception&) {
              throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: rank " + std::to_string(rank) + " :: " + "exception reading sample: " + cnames[i] + " which is " + std::to_string(i) + " of " + std::to_string(cnames[i].size()) + "; " + files[j]);
            }

            ++sample_count;
            if (sample_count == samples_per_file) {
              try {
                conduit::relay::io::save(save_me, output_fn, "hdf5");
              } catch (const std::exception& e) {
                std::cerr << rank << " :: exception: failed to save conduit node to disk; what: " << e.what() << "\n";
                continue;
              } catch (...) {
                std::cerr << rank << " :: exception: failed to save conduit node to disk\n";
                continue;
              }

              save_me.reset();
              sprintf(output_fn, "%s/%d/samples_%d_%d.bundle", base.c_str(), cur_dir++, cur_file++, rank);
              sample_count = 0;
              if (cur_dir == num_dirs) {
                cur_dir = 0;
              }
            }
          }
        }
      }
      if (sample_count) {
        try {
          conduit::relay::io::save(save_me, output_fn, "hdf5");
        } catch (exception const& e) {
          std::cerr << rank << " :: exception: failed to save conduit node to disk; what: " << e.what() << "\n";
        } catch (...) {
          std::cerr << rank << " :: exception: failed to save conduit node to disk; FINAL FILE\n";
        }
      }

  } catch (std::exception const& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}
