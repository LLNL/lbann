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
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/protobuf_utils.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_DNN_LIB

#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/model.pb.h"

#include <cstdlib>

#if __has_include(<unistd.h>) && __has_include(<sys/types.h>)
#define LBANN_HAS_UNISTD_H
#include <sys/types.h>
#include <unistd.h>
#endif

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
  arg_parser.add_option("hang on start",
                        {"--hang"},
                        "[STD] Hang given MPI rank",
                        -1);

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
    std::cout << "\n\n"
              << std::string(62, '=') << '\n'
              << "STARTING lbann with this command line:\n";
    for (int j = 0; j < argc; j++) {
      std::cout << argv[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  try {
    if (arg_parser.help_requested() or argc == 1) {
      if (master)
        std::cout << arg_parser << std::endl;
      return EXIT_SUCCESS;
    }

    auto const hang_rank = arg_parser.get<int>("hang on start");
    if (hang_rank >= 0) {
      auto const my_rank = comm->get_rank_in_world();
      if (hang_rank == my_rank) {
#ifdef LBANN_HAS_UNISTD_H
        char hostname[1024];
        gethostname(hostname, 1024);
        std::cout << "LBANN [hang]: (hostname: " << hostname
                  << ", pid: " << getpid() << ")" << std::endl;
#endif
        int volatile wait = 1;
        while (wait) {
        }
      }
      comm->global_barrier();
    }

    // Setup cuDNN and cuBLAS defaults
    auto use_cudnn_tensor_ops =
      arg_parser.get<bool>(LBANN_OPTION_USE_CUDNN_TENSOR_OPS);
    auto use_cublas_tensor_ops =
      arg_parser.get<bool>(LBANN_OPTION_USE_CUBLAS_TENSOR_OPS);
    if (master) {
      std::cout << "Default tensor core settings:\n"
                << "   cuDNN: " << (use_cudnn_tensor_ops ? "" : "NOT ")
                << "using tensor core math."
                << "\n"
                << "  cuBLAS: " << (use_cublas_tensor_ops ? "" : "NOT ")
                << "using tensor core math."
                << "\n"
                << std::endl;
    }
#ifdef LBANN_HAS_DNN_LIB
    if (use_cudnn_tensor_ops)
      dnn_lib::default_to_tensor_ops();
#endif // LBANN_HAS_DNN_LIB
#ifdef LBANN_HAS_CUDA
    if (use_cublas_tensor_ops)
      El::gpu_blas::RequestTensorOperations();
#endif // LBANN_HAS_CUDA

    // this must be called after call to arg_parser.parse();
    if (!arg_parser.get<bool>(LBANN_OPTION_DISABLE_SIGNAL_HANDLER)) {
      std::string file_base =
        (arg_parser.get<bool>(LBANN_OPTION_STACK_TRACE_TO_FILE) ? "stack_trace"
                                                                : "");
      stack_trace::register_signal_handler(file_base);
    }

    // Split MPI into trainers
    allocate_trainer_resources(comm.get());

    int trainer_rank = 0;
    if (arg_parser.get<bool>(LBANN_OPTION_GENERATE_MULTI_PROTO)) {
      trainer_rank = comm->get_trainer_rank();
    }
    // Load the prototexts specificed on the command line
    auto pbs = protobuf_utils::load_prototext(master, trainer_rank);
    // Optionally over-ride some values in the prototext for each model
    for (size_t i = 0; i < pbs.size(); i++) {
      get_cmdline_overrides(*comm, *(pbs[i]));
    }

    lbann_data::LbannPB& pb = *(pbs[0]);
    lbann_data::Trainer* pb_trainer = pb.mutable_trainer();

    // Construct the trainer
    auto& trainer = construct_trainer(comm.get(), pb_trainer, pb);

    thread_pool& io_thread_pool = trainer.get_io_thread_pool();

    lbann_data::Model* pb_model = pb.mutable_model();

    auto model =
      build_model_from_prototext(argc,
                                 argv,
                                 pb_trainer,
                                 pb,
                                 comm.get(),
                                 io_thread_pool,
                                 trainer.get_callbacks_with_ownership());

    if (!arg_parser.get<bool>(LBANN_OPTION_EXIT_AFTER_SETUP)) {

      // Train model
      trainer.train(model.get(), pb_model->num_epochs());

      // Evaluate model on test set
      trainer.evaluate(model.get(), execution_mode::testing);
    }
    else {
      if (comm->am_world_master()) {
        std::cout
          << "-----------------------------------------------------------------"
             "---------------\n"
             "ALERT: model has been setup; we are now exiting due to command\n"
             "       line option: --exit_after_setup\n"
             "-----------------------------------------------------------------"
             "---------------\n";
      }
    }
  }
  catch (lbann::exception& e) {
    if (arg_parser.get<bool>(LBANN_OPTION_STACK_TRACE_TO_FILE)) {
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
  }
  catch (std::exception& e) {
    El::ReportException(e);
    // It's possible that a proper subset of ranks throw some
    // exception. But we want to tear down the whole world.
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
