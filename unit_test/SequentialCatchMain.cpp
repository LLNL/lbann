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
#include <lbann/utils/dnn_lib/helpers.hpp>
#include <lbann/utils/random_number_generators.hpp>

int main(int argc, char* argv[]) {
#ifdef LBANN_HAS_DNN_LIB
  hydrogen::gpu::Initialize();
  dnn_lib::initialize();
#endif // LBANN_HAS_DNN_LIB


  // Initialize the general RNGs and the data sequence RNGs
  int random_seed = 42;
  lbann::init_random(random_seed);
  lbann::init_data_seq_random(random_seed);

  int result = Catch::Session().run(argc, argv);


#ifdef LBANN_HAS_DNN_LIB
  lbann::dnn_lib::destroy();
  hydrogen::gpu::Finalize();
#endif // LBANN_HAS_DNN_LIB

  return result;
}
