////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);
    if (opts->has_string("h") or opts->has_string("help") or argc == 1) {
      print_help(comm);
      finalize(comm);
      return 0;
    }

    std::stringstream err;

    std::vector<lbann_data::LbannPB *> pbs;
    protobuf_utils::load_prototext(master, argc, argv, pbs);

    model *model_1 = build_model_from_prototext(argc, argv, *(pbs[0]),
                                                comm, true); //ae
    model *model_2 = nullptr; //cycgan
    model *model_3 = nullptr; //ae+cycgan


    if (pbs.size() > 1) {
      model_2 = build_model_from_prototext(argc, argv, *(pbs[1]),
                                           comm, false);
    }

    if (pbs.size() > 2) {
      model_3 = build_model_from_prototext(argc, argv, *(pbs[2]),
                                           comm, false);
    }


    const lbann_data::Model pb_model_1 = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();
    const lbann_data::Model pb_model_3 = pbs[2]->model();

    if(master) std::cout << " Pre-train autoencoder " << std::endl;
    model_1->train(pb_model_1.num_epochs());
    model_1->evaluate(execution_mode::testing);
    auto ae_weights = model_1->get_weights();
    model_2->copy_trained_weights_from(ae_weights);
    model_3->copy_trained_weights_from(ae_weights);

    //Train cycle GAN
    if (master)  std::cerr << "\nSTARTING train - cycle GAN \n\n";
    model_2->train(pb_model_2.num_epochs());
    model_2->evaluate(execution_mode::testing);
    auto model2_weights = model_2->get_weights();
      
    //Evaluate on pretrained autoencoder
    if(master) std::cout << " Copy trained weights from cycle GAN" << std::endl;
    model_3->copy_trained_weights_from(model2_weights);
    if(master) std::cout << " Evaluate pretrained autoencoder" << std::endl;
    model_3->evaluate(execution_mode::testing);

    delete model_1;
    if (model_2 != nullptr) {
      delete model_2;
    }
    if (model_3 != nullptr) {
      delete model_3;
    }
    for (auto t : pbs) {
      delete t;
    }

  } catch (std::exception& e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  }

  // Clean up
  finalize(comm);
  return EXIT_SUCCESS;
}
