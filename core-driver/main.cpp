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
#include "lbann/utils/threads/thread_utils.hpp"
#include <mpi.h>
#include <stdio.h>

//#include <thread>
//#include <chrono>

std::unique_ptr<lbann::directed_acyclic_graph_model>
load_model(lbann::lbann_comm* lc, std::string cp_loc) {
  // Data Coordinator
  std::unique_ptr<lbann::data_coordinator> dc;
  dc = lbann::make_unique<lbann::buffered_data_coordinator<float>>(lc);

  // Trainer
  int mbs = 64;
  auto t = lbann::make_unique<lbann::trainer>(lc, mbs, std::move(dc));

  // Checkpoint location
  auto& p = t->get_persist_obj();
  p.open_restart(cp_loc.c_str());

  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, lbann::free_core_offset(lc));

  // Datareader, but this will go away
  lbann::init_data_seq_random(-1);
  std::map<lbann::execution_mode, lbann::generic_data_reader *> data_readers;
  lbann::generic_data_reader *reader = nullptr;
  reader = new lbann::mnist_reader(false);
  reader->set_comm(lc);
  reader->set_data_filename("t10k-images-idx3-ubyte");
  reader->set_label_filename("t10k-labels-idx1-ubyte");
  reader->set_file_dir("/usr/WS2/wyatt5/pascal/lbann-save-model/applications/vision/data/mnist");
  reader->set_role("test");
  reader->set_master(lc->am_world_master());
  reader->load();
  data_readers[lbann::execution_mode::testing] = reader;

  // Load trainer from checkpoint
  auto t_flag = t->load_from_checkpoint_shared(p);
  std::cout << "trainer load: " << t_flag << std::endl;
  t->setup(std::move(io_thread_pool), data_readers);

  // Model
  auto m = lbann::make_unique<lbann::directed_acyclic_graph_model>(lc, nullptr, nullptr);

  // Load model from checkpoint
  auto m_flag = m->load_from_checkpoint_shared(p);
  std::cout << "model load: " << m_flag << std::endl;

  p.close_restart();

  return m;
}

void load_samples(std::string sample_loc) {

}

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
  }

  // Setup comms
  std::cout << "initializing LBANN..." << std::endl;
  auto lbann_comm = lbann::driver_init(MPI_COMM_WORLD);

  int ppt = lbann_comm->get_procs_in_world();
  if (ppt != lbann_comm->get_procs_per_trainer()) {
    lbann_comm->split_trainers(ppt);
  }

  // Load the model
  std::string model_dir;
  model_dir = "/usr/workspace/wyatt5/cp_models/trainer0/sgd.shared.epoch_begin.epoch.10.step.8440/";
  auto m = load_model(lbann_comm.get(), model_dir);

  // Load the data
  std::string sample_dir;
  sample_dir = "/usr/workspace/wyatt5/mnist_data/mnist.csv";
  //auto samples = load_data(sample_dir);

  // Infer
  //auto inf = infer(m.get(), samples.key, samples.values);

  // Clean up
  m.reset();
  lbann::finalize();
  MPI_Finalize();

  return 0;
}
