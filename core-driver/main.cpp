///////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//// Produced at the Lawrence Livermore National Laboratory.
//// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
//// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
////
//// LLNL-CODE-697807.
//// All rights reserved.
////
//// This file is part of LBANN: Livermore Big Artificial Neural Network
//// Toolkit. For details, see http://software.llnl.gov/LBANN or
//// https://github.com/LLNL/LBANN.
////
//// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
//// may not use this file except in compliance with the License.  You may
//// obtain a copy of the License at:
////
//// http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//// implied. See the License for the specific language governing
//// permissions and limitations under the license.
///////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
  }

  std::cout << "initializing LBANN..." << std::endl;
  auto lbann_comm = lbann::driver_init(MPI_COMM_WORLD);
  lbann::finalize();
  std::cout << "LBANN initialized." << std::endl;

  MPI_Finalize();
  return 0;
}
