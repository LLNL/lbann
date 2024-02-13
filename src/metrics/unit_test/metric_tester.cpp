////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

// Example metric executable for use in the executable_metric unit tests

#include <iostream>

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 3) {
    std::cout << "USAGE: "
              << "metric-tester [extra argument] <trainer and model ID>"
              << std::endl;
    return 1;
  }

  // Failing cases
  if (argc == 3) {
    if (argv[1][0] == 'f') {
      // Non-numeric output
      std::cout << "fail" << std::endl;
      return 0;
    }
    else if (argv[1][0] == 'r') {
      // Nonzero return code
      return 2;
    }
    else {
      // Different output value
      std::cout << "-2.8" << std::endl;
      return 0;
    }
  }

  // Successful run
  std::cout << "1.40000" << std::endl;

  return 0;
}
