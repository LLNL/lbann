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
#include "lbann/weights/data_type_weights.hpp"
#include <mpi.h>
#include <stdio.h>

//#include <thread>
//#include <chrono>

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
  }

  // Comms
  std::cout << "initializing LBANN..." << std::endl;
  auto lbann_comm = lbann::driver_init(MPI_COMM_WORLD);

  int ppt = lbann_comm->get_procs_in_world();
  if (ppt != lbann_comm->get_procs_per_trainer()) {
    lbann_comm->split_trainers(ppt);
  }

  // Data Coordinator
  std::unique_ptr<lbann::data_coordinator> dc;
  dc = lbann::make_unique<lbann::buffered_data_coordinator<float>>(lbann_comm.get());

  // Trainer
  int mbs = 64;
  auto t = lbann::make_unique<lbann::trainer>(lbann_comm.get(), mbs, std::move(dc));

  // Checkpoint location
  auto& p = t->get_persist_obj();
  std::string cp_loc;
  cp_loc = "/usr/workspace/wyatt5/cp_models/trainer0/sgd.shared.epoch_begin.epoch.10.step.8440/";
  p.open_restart(cp_loc.c_str());

  // Load trainer from checkpoint
  auto t_flag = t->load_from_checkpoint_shared(p);
  std::cout << "trainer load: " << t_flag << std::endl;

  // Model
  auto m = lbann::directed_acyclic_graph_model(lbann_comm.get(), nullptr, nullptr);

  // Load model from checkpoint
  auto m_flag = m.load_from_checkpoint_shared(p);
  std::cout << "model load: " << m_flag << std::endl;
  load_weights_from_checkpoint(m, cp_loc);

  // Clean up
  p.close_restart();
  lbann::finalize();
  MPI_Finalize();

  return 0;
}
