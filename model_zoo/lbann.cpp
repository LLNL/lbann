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
#include "lbann/utils/argument_parser.hpp"
#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/cudnn.hpp"
#endif // LBANN_HAS_CUDNN

#include <lbann.pb.h>
#include <model.pb.h>

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
}// namespace <anon>

int main(int argc, char *argv[]) {
  auto& arg_parser = global_argument_parser();
  construct_std_options();
  auto use_cudnn_tensor_ops =
    arg_parser.add_flag("use cudnn tensor ops",
                        {"--use-cudnn-tensor-ops"},
                        utils::ENV("LBANN_USE_CUDNN_TENSOR_OPS"),
                        "Set the default cuDNN math mode to use "
                        "Tensor Core operations when available.");
  auto use_cublas_tensor_ops =
    arg_parser.add_flag("use cublas tensor ops",
                        {"--use-cublas-tensor-ops"},
                        utils::ENV("LBANN_USE_CUBLAS_TENSOR_OPS"),
                        "Set the default cuBLAS math mode to use "
                        "Tensor Core operations when available.");
  try {
    arg_parser.parse(argc, argv);
  }
  catch (std::exception const& e) {
    auto guessed_rank = guess_global_rank();
    if (guessed_rank <= 0)
      // Cannot call `El::ReportException` because MPI hasn't been
      // initialized yet.
      std::cerr << "Error during argument parsing:\n\ne.what():\n\n  "
                << e.what() << "\n\nProcess terminating."
                << std::endl;
    std::terminate();
  }

  world_comm_ptr comm = initialize(argc, argv);
  const bool master = comm->am_world_master();

  if (master) {
    std::cout << "\n\n" << std::string(62,'=') << '\n'
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
      if (master)
        std::cout << arg_parser << std::endl;
      print_help(*comm);
      return EXIT_SUCCESS;
    }

    // Setup cuDNN and cuBLAS defaults
    if (master) {
      std::cout << "Default tensor core settings:\n"
                << "   cuDNN: " << (use_cudnn_tensor_ops ? "" : "NOT ")
                << "using tensor core math." << "\n"
                << "  cuBLAS: " << (use_cublas_tensor_ops ? "" : "NOT ")
                << "using tensor core math." << "\n"
                << std::endl;
    }
#ifdef LBANN_HAS_CUDNN
    if (use_cudnn_tensor_ops)
      cudnn::default_to_tensor_ops();
#endif // LBANN_HAS_CUDNN
#ifdef LBANN_HAS_CUDA
    if (use_cublas_tensor_ops)
      El::gpu_blas::RequestTensorOperations();
#endif // LBANN_HAS_CUDA

    //this must be called after call to opts->init();
    if (!opts->get_bool("disable_signal_handler")) {
      std::string file_base = (opts->get_bool("stack_trace_to_file") ?
                               "stack_trace" : "");
      stack_trace::register_signal_handler(file_base);
    }

    //to activate, must specify --st_on on cmd line
    stack_profiler::get()->activate(comm->get_rank_in_world());

    // Load the prototexts specificed on the command line
    auto pbs = protobuf_utils::load_prototext(master, argc, argv);
    // Optionally over-ride some values in the prototext for each model
    for(size_t i = 0; i < pbs.size(); i++) {
      get_cmdline_overrides(*comm, *(pbs[i]));
    }

    lbann_data::LbannPB& pb = *(pbs[0]);
    lbann_data::Trainer *pb_trainer = pb.mutable_trainer();

    // Construct the trainer
    std::unique_ptr<trainer> trainer = construct_trainer(comm.get(), pb_trainer, pb, opts);

    thread_pool& io_thread_pool = trainer->get_io_thread_pool();

    int training_dr_linearized_data_size = -1;
    auto *dr = trainer->get_data_coordinator().get_data_reader(execution_mode::training);
    if(dr != nullptr) {
      training_dr_linearized_data_size = dr->get_linearized_data_size();
    }

    lbann_data::Model *pb_model = pb.mutable_model();

    auto model = build_model_from_prototext(argc, argv, pb_trainer, pb,
                                            comm.get(), opts, io_thread_pool,
                                            trainer->get_callbacks_with_ownership(),
                                            training_dr_linearized_data_size);

    if (opts->has_string("create_tarball")) {
      return EXIT_SUCCESS;
    }

    if (! opts->get_bool("exit_after_setup")) {

      // Train model
      trainer->train(model.get(), pb_model->num_epochs());

      // Evaluate model on test set
      trainer->evaluate(model.get(), execution_mode::testing);

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
    // It's possible that a proper subset of ranks throw some
    // exception. But we want to tear down the whole world.
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  } catch (std::exception& e) {
    El::ReportException(e);
    // It's possible that a proper subset of ranks throw some
    // exception. But we want to tear down the whole world.
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
