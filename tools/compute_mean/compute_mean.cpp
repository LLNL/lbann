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

#include <iostream>
#include "mpi_states.hpp"
#include "walltimes.hpp"
#include "params.hpp"
#include "image_list.hpp"
#include "process_images.hpp"
#include "lbann/utils/random.hpp"

int main(int argc, char *argv[]) {
  using namespace tools_compute_mean;

  mpi_states ms;
  ms.initialize(argc, argv);

  walltimes wt;
  params mp;

  // Parse the command line arguments
  bool ok = mp.set(argc, argv);
  if (!ok) {
    if (ms.is_root()) {
      std::cout << mp.show_help(argv[0]) << std::endl;
    }
    ms.finalize();
    return 0;
  }
  mp.set_out_ext(".JPEG");
  lbann::init_random(mp.get_seed() + ms.get_my_rank());

  // Load the image list
  image_list img_list(mp.get_data_path_file(), mp.to_write_cropped(), ms);
  if (mp.check_to_create_dirs_only()) return 0;

  // Check the effective number of ranks which have data to process
  ms.set_effective_num_ranks(img_list.get_effective_num_ranks());
  if (ms.get_effective_num_ranks() == 0) {
    ms.abort("No image to process!");
  }
  if (ms.is_root() && (ms.get_effective_num_ranks() != ms.get_num_ranks())) {
    std::cerr << "Number of effective ranks: " << ms.get_effective_num_ranks() << std::endl;
  }

  // The main loop of processing images and extracting the mean
  ok = process_images(img_list, mp, ms, wt);
  if (!ok) {
    ms.abort("Failed!");
  }

  // Summarize timing measurements
  summarize_walltimes(wt, ms);

  ms.finalize();

  return 0;
}
