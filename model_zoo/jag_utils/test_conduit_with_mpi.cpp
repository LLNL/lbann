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
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  int np = comm->get_procs_in_world();

  if (master) {
    if (np != 2) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: please run with exactly two processes");
    }
  }

  try {
    int rank = comm->get_rank_in_world();
    conduit::Node n;

    if (rank == 0) {
      n["some data"] = 42;
      n["some more data"] = "it's turtles, all the way down";
      std::cout << "P_0 is sending the following conduit node to P_1:\n";
      n.print();
    }
    comm->global_barrier();

    if (rank == 0) {
      conduit::relay::mpi::send_using_schema(n, 1, 22, MPI_COMM_WORLD);
    }

    else {
      conduit::Node n2;
      conduit::relay::mpi::recv_using_schema(n2, 0, 22, MPI_COMM_WORLD);
      std::cout << "\nP_1 received the following node from P_0:\n";
      n2.print();
    }

  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;

#endif //if 0
}

#endif //#ifdef LBANN_HAS_CONDUIT
