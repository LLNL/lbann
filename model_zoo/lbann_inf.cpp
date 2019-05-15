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
#include <dirent.h>
#include <cstdlib>
using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  auto comm = initialize(argc, argv, random_seed);
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
    std::shared_ptr<thread_pool> io_thread_pool
      = construct_io_thread_pool(comm.get());

    auto pbs = protobuf_utils::load_prototext(master, argc, argv);
    std::vector<std::unique_ptr<model>> models;
    for(auto&& pb_model : pbs) {
      models.emplace_back(
        build_model_from_prototext(argc, argv, *pb_model,
                                   comm.get(), io_thread_pool, models.size() == 0));
    }

    // Load layer weights from checkpoint if checkpoint directory given
    if(opts->has_string("ckpt_dir")){
      for(auto&& m : models) {
        lbann_callback_save_model::load_model_weights(opts->get_string("ckpt_dir"), m.get());
      }
    }else {
      LBANN_ERROR("Unable to reload model");
    }

    /// Interleave the inference between the models so that they can use a shared data reader
    /// Enable shared testing data readers on the command line via --share_testing_data_readers=1
    El::Int num_samples = models[0]->get_num_iterations_per_epoch(execution_mode::testing);
    for(El::Int s = 0; s < num_samples; s++) {
      for(auto&& m : models) {
        m->evaluate(execution_mode::testing, 1);
      }
    }

  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
