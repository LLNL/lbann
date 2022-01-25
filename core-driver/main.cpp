///////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/argument_parser.hpp"
//#include <conduit/conduit.hpp>
#include <mpi.h>
#include <stdio.h>

void construct_opts(int argc, char **argv) {
  auto& arg_parser = lbann::global_argument_parser();
  lbann::construct_std_options();
  lbann::construct_datastore_options();
  arg_parser.add_option("samples",
                        {"-n"},
                        "Number of samples to run inference on",
                        128);
  arg_parser.add_option("channels",
                        {"-c"},
                        "Number of image channels in sample",
                        1);
  arg_parser.add_option("height",
                        {"-h"},
                        "Height of image in sample",
                        28);
  arg_parser.add_option("width",
                        {"-w"},
                        "Width of image in sample",
                        28);
  arg_parser.add_option("labels",
                        {"-l"},
                        "Number of labels in dataset",
                        10);
  arg_parser.add_option("minibatchsize",
                        {"-mbs"},
                        "Number of samples in a mini-batch",
                        16);
  arg_parser.add_required_argument<std::string>
                                  ("model",
                                   "Directory containing checkpointed model");
  arg_parser.parse(argc, argv);
}

El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>
random_samples(El::Grid const& g, int n, int c, int h, int w) {
  El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>
    samples(n, c * h * w, g);
  El::MakeUniform(samples);
  return samples;
}

void random_mnist_sample(int *arr, int size, int max_val=255) {
  for (int i; i < size; i++) {
    arr[i] = std::rand() % max_val;
  }
}

std::vector<conduit::Node> conduit_mnist_samples(int n, int c, int h, int w) {
  std::vector<conduit::Node> samples(n);
  int sample_size = c * h * w;
  for (int i; i<n; i++) {
    int this_sample[sample_size];
    random_mnist_sample(this_sample, sample_size);
    samples[i]["data/samples"].set(this_sample, sample_size);
    samples[i]["data/labels"] = std::rand() % 10;
  }
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

  // Get input arguments and print values
  construct_opts(argc, argv);
  auto& arg_parser = lbann::global_argument_parser();
  if (rank == 0) {
    std::stringstream msg;
    msg << "Model: " << arg_parser.get<std::string>("model") << std::endl;
    msg << "{ N, c, h, w } = { " << arg_parser.get<int>("samples") << ", ";
    msg << arg_parser.get<int>("channels") << ", ";
    msg << arg_parser.get<int>("height") << ", ";
    msg << arg_parser.get<int>("width") << " }" << std::endl;
    msg << "label count = " << arg_parser.get<int>("labels") << std::endl;
    msg << "max MBS = " << arg_parser.get<int>("minibatchsize") << std::endl;
    std::cout << msg.str();
  }

  // Load model and run inference on samples
  auto lbann_comm = lbann::initialize_lbann(MPI_COMM_WORLD);

  auto samples = conduit_mnist_samples(arg_parser.get<int>("samples"),
                                       arg_parser.get<int>("channels"),
                                       arg_parser.get<int>("height"),
                                       arg_parser.get<int>("width"));
  samples[1].print();

  auto m = lbann::load_inference_model(lbann_comm.get(),
                                       arg_parser.get<std::string>("model"),
                                       samples,
                                       arg_parser.get<int>("minibatchsize"),
                                       {
                                         arg_parser.get<int>("channels"),
                                         arg_parser.get<int>("height"),
                                         arg_parser.get<int>("width")
                                       },
                                       {arg_parser.get<int>("labels")});

  auto labels = lbann::inference(m.get(), arg_parser.get<int>("minibatchsize"));

  /*
  auto samples = random_samples(lbann_comm->get_trainer_grid(),
                                arg_parser.get<int>("samples"),
                                arg_parser.get<int>("channels"),
                                arg_parser.get<int>("height"),
                                arg_parser.get<int>("width"));

  auto labels = lbann::infer(m.get(),
                             samples,
                             arg_parser.get<int>("minibatchsize"));
  */

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
