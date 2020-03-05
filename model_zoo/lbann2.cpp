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
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf_utils.hpp"

#include <lbann.pb.h>
#include <model.pb.h>

#include <dirent.h>

#include <cstdlib>
using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  const bool master = comm->am_world_master();

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);
    if (opts->has_string("h") or opts->has_string("help") or argc == 1) {
      print_help(*comm);
      return EXIT_SUCCESS;
    }

    std::ostringstream err;

    auto pbs = protobuf_utils::load_prototext(master, argc, argv);
    // Optionally over-ride some values in the prototext for each model
    for(size_t i = 0; i < pbs.size(); i++) {
      get_cmdline_overrides(*comm, *(pbs[i]));
    }

    lbann_data::LbannPB& pb = *(pbs[0]);
    lbann_data::Trainer *pb_trainer = pb.mutable_trainer();

    // Construct the trainer
    std::unique_ptr<trainer> trainer = construct_trainer(comm.get(), pb_trainer, opts);

    thread_pool& io_thread_pool = trainer->get_io_thread_pool();

    auto model_1 = build_model_from_prototext(argc, argv, pb_trainer, *(pbs[0]),
                                              comm.get(), opts, io_thread_pool,
                                              trainer->get_callbacks_with_ownership(), true);

    // Load layer weights from checkpoint if checkpoint directory given
    if(opts->has_string("ckpt_dir")){
      callback::load_model::load_model_weights(opts->get_string("ckpt_dir"),
                                               model_1.get());
    }
    // Train model
    if (master) {
      std::cerr << "\nSTARTING train - model 1\n\n";
    }
    const lbann_data::Model pb_model = pbs[0]->model();

    // When using checkpoint states, skip training as those could be the result
    // of checkpointing by steps.
    if (!opts->has_string("no_model1_train")){
      trainer->train(model_1.get(), pb_model.num_epochs() );
    }
    // Evaluate model 1 unless it is set to skip
    if (!opts->has_string("no_model1_eval")){
      trainer->evaluate(model_1.get(), execution_mode::testing);
    }


    std::unique_ptr<model> model_2;
    if (pbs.size() > 1) {
      // Reset the RNGs
      init_random(random_seed);
      init_data_seq_random(random_seed);
      model_2 = build_model_from_prototext(argc, argv, pb_trainer, *(pbs[1]),
                                           comm.get(), opts, io_thread_pool,
                                           trainer->get_callbacks_with_ownership(), false);

    }
    if (model_2 != nullptr) {
      const auto layers1 = model_1->get_layers();
      const auto layers2 = model_2->get_layers();
      for(size_t l2=0; l2 < layers2.size(); l2++) {
        for(size_t l1=0; l1 < layers1.size(); l1++) {
           if(layers2[l2]->get_name() == layers1[l1]->get_name()){
             if(master) {
               std::cout << "Model 1 Layer " << layers1[l1]->get_name();
             }
             layers2[l2]->replace_weights(layers1[l1]);
             if(master) {
               std::cout << " copied to Model2 Layer " << std::endl;
             }
           }
         }
       }

      if (master) {
        std::cerr << "\n STARTING train - model 2\n\n";
      }
      const lbann_data::Model pb_model_2 = pbs[1]->model();
      trainer->train(model_2.get(), pb_model_2.num_epochs() );
      trainer->evaluate(model_2.get(), execution_mode::testing);
    }

  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
