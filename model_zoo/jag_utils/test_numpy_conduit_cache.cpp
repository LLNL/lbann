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
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/data_readers/numpy_conduit_cache.hpp"

using namespace lbann;

// sample code that demonstrates use of the numpy_conduit_cache class

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " numpy_npz_file_name\n"
              << "Run with a single processor\n";
    exit(9);
  }

  try {

    numpy_conduit_cache n(comm.get());
    n.load(argv[1], 42);
  
    conduit::Node cosmo_base = n.get_conduit_node(42);
  
    conduit::Node cosmo = cosmo_base["42"];
    auto children = cosmo.children();
    while (children.has_next()) {
      conduit::Node &child = children.next();
      size_t word_size = child["word_size"].value();
      size_t num_vals = child["num_vals"].value();
      auto shape = child["shape"].as_uint64_array();
      int shape_num_elts = shape.number_of_elements();
      char *data = child["data"].value();
      std::cout 
        << "\nnext child: " << child.name() << "\n"
        << "  word_size: " << word_size << "\n"
        << "  num_vals:  " << num_vals << "\n"
        << "  shape:     ";
      for (int k=0; k<shape_num_elts; k++) {
        std::cout << shape[k] << " ";
      }
      std::cout << "\n  data:     " << data[0] << " ...\n";
    }
  
    /*
    std::cout << "\n=====================================================\n"
              << "cosmo.print_detailed(): \n\n";
    cosmo.print_detailed();
    */

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

#endif //#ifdef LBANN_HAS_CONDUIT
