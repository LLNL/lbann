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

#include <dirent.h>

#include <cstdlib>

using namespace lbann;

int main(int argc, char* argv[])
{
  auto& arg_parser = global_argument_parser();
  construct_all_options();

  try {
    arg_parser.parse(argc, argv);
  }
  catch (std::exception const& e) {
    std::cerr << "Error during argument parsing:\n\ne.what():\n\n  " << e.what()
              << "\n\nProcess terminating." << std::endl;
    std::terminate();
  }
  auto comm = initialize(argc, argv);
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
    auto* dr =
      trainer.get_data_coordinator().get_data_reader(execution_mode::testing);
    if (dr == nullptr) {
      LBANN_ERROR("No testing data reader defined");
    }

    std::vector<std::unique_ptr<model>> models;
    for (auto&& pb_model : pbs) {
      models.emplace_back(
        build_model_from_prototext(argc,
                                   argv,
                                   pb_trainer,
                                   *pb_model,
                                   comm.get(),
                                   io_thread_pool,
                                   trainer.get_callbacks_with_ownership()));
    }

    /// Interleave the inference between the models so that they can use a
    /// shared data reader Enable shared testing data readers on the command
    /// line via --share_testing_data_readers=1
    El::Int num_samples = dr->get_num_iterations_per_epoch();
    if (num_samples == 0) {
      LBANN_ERROR("The testing data reader does not have any samples");
    }
    for (El::Int s = 0; s < num_samples; s++) {
      for (auto&& m : models) {
        trainer.evaluate(m.get(), execution_mode::testing, 1);
      }
    }
  }
  catch (std::exception& e) {
    El::ReportException(e);
    // It's possible that a proper subset of ranks throw some
    // exception. But we want to tear down the whole world.
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
