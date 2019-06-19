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

    // Initalize a global I/O thread pool
    std::shared_ptr<thread_pool> io_thread_pool = construct_io_thread_pool(comm.get());

    auto pbs = protobuf_utils::load_prototext(master, argc, argv);

    auto model_1 = build_model_from_prototext(argc, argv, *(pbs[0]), comm.get(), io_thread_pool, true); //discriminator
                                                                                    //model
    std::unique_ptr<model> model_2 = nullptr; //adversarial model
    if (pbs.size() > 1) {
      model_2 = build_model_from_prototext(argc, argv, *(pbs[1]), comm.get(), io_thread_pool, false);
    }

    const lbann_data::Model pb_model = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();

    const auto layers1 = model_1->get_layers();
    const auto layers2 = model_2->get_layers();
    int super_step = 1;
    int max_super_step = pb_model.super_steps();
    while (super_step <= max_super_step) {
      if (master)  std::cerr << "\nSTARTING train - discriminator model at step " << super_step <<"\n\n";
      //@todo freeze generator layers in this step
      model_1->train( super_step*pb_model.num_epochs() );

      //Replace/copy "proxy" layer in adversarial model (model2) with its "equivalent" layer in discriminator model (model1)
      //@todo freeze layers after replacement
      for(size_t l2=0; l2 < layers2.size(); l2++) {
        //check if a discriminator layer is a proxy
        std::string l2_fullname = layers2[l2]->get_name();
        if(l2_fullname.find("proxy") != std::string::npos) { //if a proxy layer
          std::string l2_name = l2_fullname.erase(l2_fullname.length()-6);
          std::cout << "L2 Name " << l2_name << std::endl;
          for(size_t l1=0; l1 < layers1.size(); l1++) {
             if(l2_name == layers1[l1]->get_name()){
               if(master) std::cout << "Replacing adversarial model (model 2) Layer " << layers1[l1]->get_name();
               layers2[l2]->replace_weights(layers1[l1]);
               if(master) std::cout << " with corresponding layer " << layers2[l2]->get_name() << " in discriminator model (model1) " << std::endl;
             }
          }
        }
      }

      if (master) std::cerr << "\n STARTING train - adversarial model at step " << super_step << " \n\n";
      model_2->train( super_step*pb_model_2.num_epochs() );

      super_step++;
    }

  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
