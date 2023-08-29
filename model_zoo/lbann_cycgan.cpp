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

  if (master) {
    std::cout
      << "\n\n==============================================================\n"
      << "STARTING lbann with this command line:\n";
    for (int j = 0; j < argc; j++) {
      std::cout << argv[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  try {
    // Split MPI into trainers
    allocate_trainer_resources(comm.get());

    if (arg_parser.help_requested() or argc == 1) {
      if (master)
        std::cout << arg_parser << std::endl;
      return EXIT_SUCCESS;
    }

    if (!arg_parser.get<bool>(LBANN_OPTION_DISABLE_SIGNAL_HANDLER)) {
      std::string file_base =
        (arg_parser.get<bool>(LBANN_OPTION_STACK_TRACE_TO_FILE) ? "stack_trace"
                                                                : "");
      stack_trace::register_signal_handler(file_base);
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

    int training_dr_linearized_data_size = -1;
    auto* dr =
      trainer.get_data_coordinator().get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      training_dr_linearized_data_size = dr->get_linearized_data_size();
    }

    auto model_1 =
      build_model_from_prototext(argc,
                                 argv,
                                 pb_trainer,
                                 *(pbs[0]),
                                 comm.get(),
                                 io_thread_pool,
                                 trainer.get_callbacks_with_ownership(),
                                 training_dr_linearized_data_size); // D1 solver
    // hack, overide model name to make reporting easy, what can break?"
    std::unique_ptr<model> model_2, // G1 solver
      model_3,                      // G2 solver

      // Support for autoencoder models
      ae_model,
      ae_cycgan_model; // contain layer(s) from (cyc)GAN

    if (pbs.size() > 1) {
      model_2 =
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *(pbs[1]),
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership(),
                                   training_dr_linearized_data_size);
    }

    if (pbs.size() > 2) {
      model_3 =
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *(pbs[2]),
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership(),
                                   training_dr_linearized_data_size);
    }

    if (pbs.size() > 3) {
      ae_model =
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *(pbs[3]),
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership(),
                                   training_dr_linearized_data_size);
    }

    if (pbs.size() > 4) {
      ae_cycgan_model =
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *(pbs[4]),
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership(),
                                   training_dr_linearized_data_size);
    }

    const lbann_data::Model pb_model = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();
    const lbann_data::Model pb_model_3 = pbs[2]->model();

    // Optionally pretrain autoencoder
    //@todo: explore joint-train of autoencoder as alternative
    if (ae_model != nullptr) {
      if (master)
        std::cout << " Pre-train autoencoder " << std::endl;
      const lbann_data::Model pb_model_4 = pbs[3]->model();
      trainer.train(ae_model.get(), pb_model_4.num_epochs());
      auto ae_weights = ae_model->get_weights();
      model_1->copy_trained_weights_from(ae_weights);
      model_2->copy_trained_weights_from(ae_weights);
      model_3->copy_trained_weights_from(ae_weights);
      ae_cycgan_model->copy_trained_weights_from(ae_weights);
    }

    // Train cycle GAN
    int super_step = 1;
    int max_super_step = pb_model.super_steps();
    while (super_step <= max_super_step) {
      if (master)
        std::cerr
          << "\nSTARTING train - discriminator (D1 & D2) models at step "
          << super_step << "\n\n";
      trainer.train(model_1.get(),
                    super_step * pb_model.num_epochs(),
                    pb_model.num_batches());

      if (master)
        std::cout << " Copy all trained weights from discriminator to G1 and "
                     "train/freeze as appropriate "
                  << std::endl;
      auto model1_weights = model_1->get_weights();
      model_2->copy_trained_weights_from(model1_weights);
      if (master)
        std::cerr << "\n STARTING train - G1 solver model at step "
                  << super_step << " \n\n";
      trainer.train(model_2.get(),
                    super_step * pb_model_2.num_epochs(),
                    pb_model_2.num_batches());
      // Evaluate model on test set
      //      model_2->evaluate(execution_mode::testing,pb_model_2.num_batches());

      if (master)
        std::cout << " Copy all trained weights from discriminator to G2 and "
                     "train/freeze as appropriate "
                  << std::endl;
      model_3->copy_trained_weights_from(model1_weights);
      if (master)
        std::cerr << "\n STARTING train - G2 solver model at step "
                  << super_step << " \n\n";
      trainer.train(model_3.get(),
                    super_step * pb_model_3.num_epochs(),
                    pb_model_3.num_batches());
      // Evaluate model on test set
      //      model_3->evaluate(execution_mode::testing,pb_model_3.num_batches());

      if (master)
        std::cout << " Update G1 weights " << std::endl;
      auto model2_weights = model_2->get_weights();
      model_1->copy_trained_weights_from(model2_weights);
      if (master)
        std::cout << " Update G2 weights " << std::endl;
      auto model3_weights = model_3->get_weights();
      model_1->copy_trained_weights_from(model3_weights);

      // Optionally evaluate on pretrained autoencoder
      if (ae_model != nullptr && ae_cycgan_model != nullptr) {
        // if(master) std::cout << " Copy trained weights from autoencoder to
        // autoencoder proxy" << std::endl;
        // ae_cycgan_model->copy_trained_weights_from(ae_weights);
        if (master)
          std::cout << " Copy trained weights from cycle GAN" << std::endl;
        ae_cycgan_model->copy_trained_weights_from(model2_weights);
        if (master)
          std::cout << " Evaluate pretrained autoencoder" << std::endl;
        // ae_cycgan_model->evaluate(execution_mode::testing);
      }

      super_step++;
    }

    model_1->save_model();
    model_2->save_model();
    model_3->save_model();
    ae_cycgan_model->save_model();
    if (master)
      std::cout << " Evaluate pretrained autoencoder" << std::endl;
    trainer.evaluate(ae_cycgan_model.get(), execution_mode::testing);
  }
  catch (std::exception& e) {
    El::ReportException(e);
    // It's possible that a proper subset of ranks throw some
    // exception. But we want to tear down the whole world.
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
