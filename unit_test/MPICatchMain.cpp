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
////////////////////////////////////////////////////////////////////////////////

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

// Utilities
#include "MPITestHelpers.hpp"
#include "ReplaceEscapes.hpp"

#include <lbann/base.hpp>
#include <lbann/utils/random_number_generators.hpp>
#include <lbann/utils/system_info.hpp>
#include <lbann/utils/options.hpp>

// Just stand up MPI before running all tests; teardown after.
using namespace unit_test::utilities;
int main(int argc, char* argv[])
{
  // Set up the communication domain
  auto world_comm = lbann::initialize(argc, argv);
  lbann::init_random(13);
  expert::register_world_comm(*world_comm);

  // as of Mar 2021, required for data_readers
  lbann::options::get()->init(argc, argv);

  // Initialize Catch2
  Catch::Session session;

  // Parse the command line
  int return_code = session.applyCommandLine(argc, argv);
  if (return_code != 0) // Indicates a command line error
    return return_code;

  // Manipulate output file if needed.
  auto& config_data = session.configData();
  auto& output_file = config_data.outputFilename;
  if (output_file.size() > 0)
  {
    lbann::utils::SystemInfo sys_info;
    output_file = replace_escapes(output_file, sys_info);
  }

  // Run the catch tests, outputting to the given file.
  int num_failed = session.run();

  // Clean up the catch environment
  expert::reset_world_comm();

  // Shut down the communication domain
  world_comm.reset(); // Force MPI_Finalize, et al, before return.

  return num_failed;
}
