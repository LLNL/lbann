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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/lbann_random.hpp"

namespace {
// Random number generator, file-visible only.
lbann::rng_gen generator;
}

namespace lbann {

rng_gen& get_generator() {
  return ::generator;
}

void init_random(int seed) {
  if (seed != -1) {
    get_generator().seed(seed);
#ifdef LBANN_SET_EL_RNG
    El::Generator().seed(seed);
#endif
  } else {
    // Seed with a random value.
    std::random_device rd;
    unsigned rand_val = rd();
    get_generator().seed(rand_val);
#ifdef LBANN_SET_EL_RNG
    El::Generator().seed(rand_val);
#endif
  }
}

}  // namespace lbann

