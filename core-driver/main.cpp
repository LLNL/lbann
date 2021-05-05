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

El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>
random_samples(El::Grid const& g, int n, int c, int h, int w) {
  El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU> samples(n, c * h * w, g);
  El::MakeUniform(samples);
  return samples;
}

int main(int argc, char **argv) {
  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Input params
  int n=128, c=1, h=28, w=28; // input dims
  int l=10; // number of labels in dataset
  int mbs = 16; // max mini-batch size
  std::string model_dir; // location of model checkpoint

  int opt;
  while ((opt = getopt (argc, argv, "n:c:h:w:l:m:d:")) != -1)
    switch (opt)
    {
      case 'n':
        n = std::atoi(optarg);
        break;
      case 'c':
        c = std::atoi(optarg);
        break;
      case 'h':
        h = std::atoi(optarg);
        break;
      case 'w':
        w = std::atoi(optarg);
        break;
      case 'l':
        l = std::atoi(optarg);
        break;
      case 'm':
        mbs = std::atoi(optarg);
        break;
      case 'd':
        model_dir = optarg;
        break;
      case '?':
        std::cerr << "Error: Unknown option -" << optopt << std::endl;
      default:
        abort();
    }
  if (model_dir.empty()) {
    if (rank == 0) {
      std::cerr << "-d requires model checkpoint directory" << std::endl;
    }
    MPI_Finalize();
    abort();
  }
  if (rank == 0) {
    std::cout << "Running inference on model at: " << model_dir << std::endl;
    std::cout << "N=" << n << ", ";
    std::cout << "c=" << c << ", ";
    std::cout << "h=" << h << ", ";
    std::cout << "w=" << w << ", ";
    std::cout << "label count=" << l << ", ";
    std::cout << "max mini-batch size=" << mbs << "." << std::endl;
  }

  // Load model and run inference on samples
  auto lbann_comm = lbann::initialize_lbann(MPI_COMM_WORLD);
  auto m = lbann::load_inference_model(lbann_comm.get(), model_dir, mbs, {1, 28, 28}, {10});
  auto samples = random_samples(lbann_comm->get_trainer_grid(), n, c, h, w);
  auto labels = lbann::infer(m.get(), samples, mbs);

  // Print inference results
  if (lbann_comm->am_world_master()) {
    std::cout << "Predicted Labels: ";
    for (int i=0; i<labels.Height(); i++) {
      std::cout << labels(i) << " ";
    }
    std::cout << std::endl;
  }

  // Clean up
  m.reset();
  lbann::finalize_lbann();
  MPI_Finalize();

  return 0;
}
