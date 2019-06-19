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
#include "lbann/data_store/data_store_conduit.hpp"
#include <cstdlib>


using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  const bool master = comm->am_world_master();

  if (master) {
    std::cout << "\n\n==============================================================\n"
              << "STARTING lbann with this command line:\n";
    for (int j=0; j<argc; j++) {
      std::cout << argv[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);
    if (opts->has_string("h") or opts->has_string("help") or argc == 1) {
      print_help(*comm);
      return EXIT_SUCCESS;
    }

    //this must be called after call to opts->init();
    if (!opts->get_bool("disable_signal_handler")) {
      std::string file_base = (opts->get_bool("stack_trace_to_file") ?
                               "stack_trace" : "");
      stack_trace::register_signal_handler(file_base);
    }

    //to activate, must specify --st_on on cmd line
    stack_profiler::get()->activate(comm->get_rank_in_world());

    // Initalize a global I/O thread pool
    std::shared_ptr<thread_pool> io_thread_pool = construct_io_thread_pool(comm.get());

    auto pbs = protobuf_utils::load_prototext(master, argc, argv);
    lbann_data::LbannPB pb = *(pbs[0]);

    lbann_data::Model *pb_model = pb.mutable_model();

    auto model = build_model_from_prototext(argc, argv, pb,
                                            comm.get(), io_thread_pool, true);

    if (opts->has_string("create_tarball")) {
      return EXIT_SUCCESS;
    }

    if (! opts->get_bool("exit_after_setup")) {

      // Train model
      model->train(pb_model->num_epochs());

      // Evaluate model on test set
      model->evaluate(execution_mode::testing);

      //has no affect unless option: --st_on was given
      stack_profiler::get()->print();

    } else {
      if (comm->am_world_master()) {
        std::cout <<
          "--------------------------------------------------------------------------------\n"
          "ALERT: model has been setup; we are now exiting due to command\n"
          "       line option: --exit_after_setup\n"
          "--------------------------------------------------------------------------------\n";
      }

      //has no affect unless option: --st_on was given
      stack_profiler::get()->print();
    }

  } catch (exception& e) {
    if (options::get()->get_bool("stack_trace_to_file")) {
      std::ostringstream ss("stack_trace");
      const auto& rank = get_rank_in_world();
      if (rank >= 0) {
        ss << "_rank" << rank;
      }
      ss << ".txt";
      std::ofstream fs(ss.str());
      e.print_report(fs);
    }
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
