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

auto mock_dr_metadata() {
  lbann::DataReaderMetaData drmd;
  auto& md_dims = drmd.data_dims;
  md_dims[lbann::data_reader_target_mode::CLASSIFICATION] = {10};
  md_dims[lbann::data_reader_target_mode::INPUT] = {1,28,28};
  return drmd;
}

std::unique_ptr<lbann::model>
load_model(lbann::lbann_comm* lc, std::string cp_dir, int mbs) {
  lbann::persist p;
  p.open_restart(cp_dir.c_str());
  auto m = lbann::make_unique<lbann::directed_acyclic_graph_model>(lc, nullptr, nullptr);
  auto m_flag = m->load_from_checkpoint_shared(p);
  if (lc->am_world_master()) {
    std::cout << "model load: " << m_flag << std::endl;
  }
  p.close_restart();

  // Must use a mock datareader with input and output dims for setup
  auto dr_metadata = mock_dr_metadata();
  m->setup(mbs, dr_metadata);

  return m;
}

El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>
load_samples(El::Grid const& g) {
  int h = 128, w = 128, c = 1, N = 64;
  El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU> samples(N, c * h * w, g);
  El::MakeUniform(samples);
  return samples;
}

int main(int argc, char *argv[]) {
  // Input params
  std::string model_dir, sample_dir, input_data_layer, input_label_layer, pred_layer;
  model_dir = "/usr/workspace/wyatt5/cp_models/trainer0/sgd.shared.epoch_begin.epoch.10.step.8440/";
  sample_dir = "/usr/workspace/wyatt5/mnist_data/mnist.csv";
  input_data_layer = "layer1";
  input_label_layer = "layer3";
  pred_layer = "layer15";
  int mbs = 16;

  // Init MPI and verify MPI_THREADED_MULTIPLE
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
  }

  // Setup comms
  auto lbann_comm = lbann::driver_init(MPI_COMM_WORLD);

  int ppt = lbann_comm->get_procs_in_world();
  if (ppt != lbann_comm->get_procs_per_trainer()) {
    lbann_comm->split_trainers(ppt);
  }

  // Setup model, samples, & exec algorithm
  auto m = load_model(lbann_comm.get(), model_dir, mbs);
  auto samples = load_samples(lbann_comm->get_trainer_grid());
  auto inf_alg = lbann::batch_functional_inference_algorithm();

  // Infer
  auto labels = inf_alg.infer(m.get(), samples, pred_layer, mbs);
  if (lbann_comm->am_world_master()) {
    std::cout << "Predicted Labels: ";
    for (int i=0; i<labels.Height(); i++) {
      std::cout << labels(i) << " ";
    }
    std::cout << std::endl;
  }

  // Clean up
  m.reset();
  lbann::finalize();
  MPI_Finalize();

  return 0;
}
