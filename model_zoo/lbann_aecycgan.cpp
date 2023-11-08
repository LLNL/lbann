////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/protobuf_utils.hpp"

#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/model.pb.h"

#include <cstdlib>

using namespace lbann;

namespace {
int guess_global_rank() noexcept
{
  int have_mpi;
  MPI_Initialized(&have_mpi);
  if (have_mpi) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
  else {
    if (char const* slurm_flag = std::getenv("SLURM_PROCID"))
      return std::stoi(slurm_flag);
    if (char const* open_mpi_flag = std::getenv("OMPI_WORLD_COMM_RANK"))
      return std::stoi(open_mpi_flag);
    else if (char const* mv2_flag = std::getenv("MV2_COMM_WORLD_LOCAL_RANK"))
      return std::stoi(mv2_flag);
    else
      return -1;
  }
}
} // namespace

int main(int argc, char* argv[])
{
  auto& arg_parser = global_argument_parser();
  construct_all_options();

  try {
    arg_parser.parse(argc, argv);
  }
  catch (std::exception const& e) {
    auto guessed_rank = guess_global_rank();
    if (guessed_rank <= 0)
      // Cannot call `El::ReportException` because MPI hasn't been
      // initialized yet.
      std::cerr << "Error during argument parsing:\n\ne.what():\n\n  "
                << e.what() << "\n\nProcess terminating." << std::endl;
    std::terminate();
  }

  world_comm_ptr comm = initialize(argc, argv);
  const bool master = comm->am_world_master();

  try {
    // Split MPI into trainers
    allocate_trainer_resources(comm.get());

    if (arg_parser.help_requested() or argc == 1) {
      if (master)
        std::cout << arg_parser << std::endl;
      return EXIT_SUCCESS;
    }

    std::ostringstream err;

    auto pbs = protobuf_utils::load_prototext(master);
    // Optionally over-ride some values in the prototext for each model
    for (size_t i = 0; i < pbs.size(); i++) {
      get_cmdline_overrides(*comm, *(pbs[i]));
    }

    lbann_data::LbannPB& pb = *(pbs[0]);
    lbann_data::Trainer* pb_trainer = pb.mutable_trainer();

    // Construct the trainer
    auto& trainer = construct_trainer(comm.get(), pb_trainer, *(pbs[0]));

    thread_pool& io_thread_pool = trainer.get_io_thread_pool();

    auto model_1 =
      build_model_from_prototext(argc,
                                 argv,
                                 pb_trainer,
                                 *(pbs[0]),
                                 comm.get(),
                                 io_thread_pool,
                                 trainer.get_callbacks_with_ownership()); // ae
    std::unique_ptr<model> model_2, // cycgan
      model_3;                      // ae+cycgan

    if (pbs.size() > 1) {
      model_2 =
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *(pbs[1]),
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership());
    }

    if (pbs.size() > 2) {
      model_3 =
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *(pbs[2]),
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership());
    }

    const lbann_data::Model pb_model_1 = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();
    const lbann_data::Model pb_model_3 = pbs[2]->model();

    if (master)
      std::cout << " Pre-train autoencoder " << std::endl;
    trainer.train(model_1.get(), pb_model_1.num_epochs());
    trainer.evaluate(model_1.get(), execution_mode::testing);
    auto ae_weights = model_1->get_weights();
    model_2->copy_trained_weights_from(ae_weights);
    model_3->copy_trained_weights_from(ae_weights);

    // Train cycle GAN
    if (master)
      std::cerr << "\nSTARTING train - cycle GAN \n\n";
    trainer.train(model_2.get(), pb_model_2.num_epochs());
    trainer.evaluate(model_2.get(), execution_mode::testing);
    auto model2_weights = model_2->get_weights();

    // Evaluate on pretrained autoencoder
    if (master)
      std::cout << " Copy trained weights from cycle GAN" << std::endl;
    model_3->copy_trained_weights_from(model2_weights);
    if (master)
      std::cout << " Save AE + cycleGAN" << std::endl;
    model_3->save_model();
    if (master)
      std::cout << " Evaluate cycleGAN model on pretrained autoencoder"
                << std::endl;
    trainer.evaluate(model_3.get(), execution_mode::testing);
  }
  catch (std::exception& e) {
    El::ReportException(e);
    // It's possible that a proper subset of ranks throw some
    // exception. But we want to tear down the whole world.
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
