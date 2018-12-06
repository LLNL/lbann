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
#include "conduit/conduit_relay_hdf5.hpp"
#include "conduit/conduit_relay_mpi.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"

using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
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
        std::cout << "usage: " << argv[0] << " --base_dir=<string> --filelist=<string> --output_fn=<string>\n"
          "where: filelist contains a list of conduit filenames;\n"
          "       base_dir / <name from filelist> should fully specify\n"
          "       a conduit filepath\n"
          "function: constructs an index that lists number of samples\n"
          "          in each file, indices of invalid samples, etc\n";
      }
      finalize(comm);
      return EXIT_SUCCESS;
    }

    if (! (opts->has_string("filelist") && opts->has_string("output_fn") && opts->has_string("base_dir"))) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: improper invocation; run with no cmd line args for proper invocation");
    }

    const std::string input_fn = opts->get_string("filelist");
    const std::string output_fn = opts->get_string("output_fn");
    const std::string base_dir = opts->get_string("base_dir");

    int rank = comm->get_rank_in_world();
    std::stringstream ss;
    ss << output_fn << "." << rank;
    std::ofstream out(ss.str().c_str());
    if (!out.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + output_fn + " for writing");
    }
    if (master) {
      std::cerr << "writing index file: " << output_fn << "\n";
      out << base_dir << "\n";
    }

    // get list of input filenames
    std::ifstream in(input_fn.c_str());
    if (!in) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + input_fn + " for reading");
    }
    std::string line;
    std::vector<std::string> filenames;
    while (in >> line) {
      if (line.size()) {
        filenames.push_back(line);
      }
    }

    int global_num_samples = 0;
    int np = comm->get_procs_in_world();
    size_t c = 0;
    size_t total_good = 0;
    hid_t hdf5_file_hnd;
    for (size_t j=rank; j<filenames.size(); j+=np) {
      ++c;
      if (c % 10 == 0) {
        std::cerr << rank << " :: processed: " << c << " files; num valid: " << total_good << "\n";
      }
      out << filenames[j] << " ";
      std::stringstream s2;
      s2 << base_dir << "/" << filenames[j];
      try {
      hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( s2.str().c_str() );
      } catch (exception e) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << e.what() << "\n";
        continue;
      } catch (...) {
         std::cerr << "exception hdf5_open_file_for_read\n";
         continue;
      }
      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch  (exception e) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names: " << e.what() << "\n";
        continue;
      } catch (...) {
         std::cerr << "exception hdf5_group_list_child_names\n";
         continue;
      }
      size_t is_good = 0;
      size_t is_bad = 0;
      std::stringstream s5;
      conduit::Node n_ok;
      for (size_t h=0; h<cnames.size(); h++) {
        const std::string key_1 = "/" + cnames[h] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key_1, n_ok);
        } catch (exception e) {
          std::cerr << rank << " :: exception hdf5_read: " << e.what() << "\n";
        } catch (...) {
           std::cerr << "exception hdf5_read file: " << s2.str() << "; key: " << key_1 << "\n";
           continue;
        }
        int success = n_ok.to_int64();
        if (success == 1) {
          ++is_good;
          ++total_good;
        } else {
          s5 << h << " ";
          ++is_bad;
        }
      }
      global_num_samples += is_good;
      out << is_good << " " << is_bad << " " << s5.str() << "\n";
      try {
        conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
      } catch (exception e) {
        std::cerr << rank << " :: exception hdf5_close_file: " << e.what() << "\n";
        continue;
      } catch (...) {
         std::cerr << "exception hdf5_close_file\n";
         continue;
      }
    }
    out.close();
    comm->global_barrier();

    int num_samples;
    MPI_Reduce(&global_num_samples, &num_samples, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (master) {
      std::stringstream s3;
      s3 << "echo " << num_samples << " " << filenames.size() << " >  num_samples_tmp";
      if (master) std::cerr << "NUM SAMPLES: " << num_samples << "\n";
      system(s3.str().c_str());
      s3.clear();
      s3.str("");
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
      s3 << "rm -f num_samples_tmp ";
      for (int k=0; k<np; k++) {
        s3 << output_fn << "." << k << " ";
      }
      system(s3.str().c_str());
    }

  } catch (exception& e) {
    std::cerr << "caught exception, outer loop!!!!\n";
    if (options::get()->has_bool("stack_trace_to_file")) {
      std::stringstream ss("stack_trace");
      const auto& rank = get_rank_in_world();
      if (rank >= 0) { ss << "_rank" << rank; }
      ss << ".txt";
      std::ofstream fs(ss.str().c_str());
      e.print_report(fs);
    }
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }

  // Clean up
  finalize(comm);
  return EXIT_SUCCESS;
}

#endif //#ifdef LBANN_HAS_CONDUIT
