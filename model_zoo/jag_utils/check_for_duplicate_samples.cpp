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

#ifdef LBANN_HAS_CONDUIT

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
#include "lbann/utils/jag_utils.hpp"

using namespace lbann;

void get_input_names(std::unordered_set<std::string> &s);
//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  // check that we're running with a single CPU
  if (np != 1) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: apologies, this is a sequential code; please run with a single processor. Thanks for playing!");
  }

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    // sanity check invocation
    if (!opts->has_string("filelist")) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string>");
      }
    }

    // read list of conduit filenames
    std::vector<std::string> files;
    const std::string fn = opts->get_string("filelist");
    read_filelist(comm.get(), fn, files);

    std::unordered_set<std::string> input_names;
    get_input_names(input_names);

    // NOTE: this is the basis for the test! We code every sample's
    //       inputs to a concatenated string
    std::unordered_set<std::string> testme;

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;
    std::cerr << "starting the test!\n";
    for (size_t j=rank; j<files.size(); ++j) {
      std::cerr << "processing: " << j << " of " << files.size() << " files\n";

      // open the next conduit file
      try {
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (...) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: hdf5_open_file_for_read failed: " + files[j]);
      }

      // get list of samples in this file
      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (std::exception e) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: hdf5_group_list_child_names failed; " + files[j]);
      }

      // loop over the samples in the current file
      for (size_t i=0; i<cnames.size(); i++) {

        // test that the sample was successful; should always be!
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: exception reading success flag: " + files[j]);
        }
        int success = n_ok.to_int64();
        if (success != 1) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: performance/success != 1; something is seriouslly wrong! " + files[j] + " " + cnames[i]);
        }

        // get inputs for the current sample and encode them as a string
        std::stringstream ss;
        try {
          for (auto t : input_names) {
            key = cnames[i] + "/inputs/" + t;
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            double d = tmp.value();
            ss << d << " ";
          }
        } catch (...) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: exception reading input from file: " + files[j]);
        }

        // test for duplicates
        std::string the_test = ss.str();
        if (testme.find(the_test) != testme.end()) {
          std::cerr <<  "duplicate set of inputs detected!\n";
        }
        testme.insert(the_test);
      }
    }
  } catch (exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}

void get_input_names(std::unordered_set<std::string> &s) {
  s.insert("shape_model_initial_modes:(4,3)");
  s.insert("betti_prl15_trans_u");
  s.insert("betti_prl15_trans_v");
  s.insert("shape_model_initial_modes:(2,1)");
  s.insert("shape_model_initial_modes:(1,0)");
}

#endif //#ifdef LBANN_HAS_CONDUIT
