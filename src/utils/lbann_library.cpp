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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/lbann_library.hpp"

#include "lbann/proto/factories.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/callbacks/load_model.hpp"
#include "lbann/utils/argument_parser.hpp"

#include <lbann.pb.h>
#include <model.pb.h>

namespace lbann {

void construct_std_options() {
  auto& arg_parser = global_argument_parser();
  arg_parser.add_option(MAX_RNG_SEEDS_DISPLAY,
                        {"--rng_seeds_per_trainer_to_display"},
                        utils::ENV("LBANN_RNG_SEEDS_PER_TRAINER_TO_DISPLAY"),
                        "Limit how many random seeds LBANN should display "
                        "from each trainer",
                        2);
  arg_parser.add_option(NUM_IO_THREADS,
                        {"--num_io_threads"},
                        utils::ENV("LBANN_NUM_IO_THREADS"),
                        "Number of threads available to both I/O and "
                        "initial data transformations for each rank.",
                        64);
  arg_parser.add_option(NUM_TRAIN_SAMPLES,
                        {"--num_train_samples"},
                        utils::ENV("LBANN_NUM_TRAIN_SAMPLES"),
                        "Set the number of training samples to ingest.",
                        0);
  arg_parser.add_option(NUM_VALIDATE_SAMPLES,
                        {"--num_validate_samples"},
                        utils::ENV("LBANN_NUM_VALIDATE_SAMPLES"),
                        "Set the number of validate samples to ingest.",
                        0);
  arg_parser.add_option(NUM_TEST_SAMPLES,
                        {"--num_test_samples"},
                        utils::ENV("LBANN_NUM_TEST_SAMPLES"),
                        "Set the number of testing samples to ingest.",
                        0);
  arg_parser.add_flag(ALLOW_GLOBAL_STATISTICS,
                      {"--ltfb_allow_global_statistics"},
                      utils::ENV("LBANN_LTFB_ALLOW_GLOBAL_STATISTICS"),
                      "Allow the print_statistics callback to report "
                      "global (inter-trainer) summary statistics.");
  arg_parser.add_option(PROCS_PER_TRAINER,
                        {"--procs_per_trainer"},
                        utils::ENV("LBANN_PROCS_PER_TRAINER"),
                        "Number of MPI ranks per LBANN trainer, "
                        "If the field is not set (or set to 0) then "
                        " all MPI ranks are assigned to one trainer."
                        " The number of processes per trainer must "
                        " evenly divide the total number of MPI ranks. "
                        " The number of resulting trainers is "
                        " num_procs / procs_per_trainer.",
                        0);
}

/// Split the MPI communicator into trainers
/// Return the
int allocate_trainer_resources(lbann_comm *comm) {
  auto& arg_parser = global_argument_parser();
  int procs_per_trainer = arg_parser.get<int>(PROCS_PER_TRAINER);

  if (procs_per_trainer == 0) {
    procs_per_trainer = comm->get_procs_in_world();
  }

  // Set up the communicator and get the grid based on the commandline spec.
  // We do not currently support splitting different trainers in different ways,
  // as this implies different grids.
  if (procs_per_trainer != comm->get_procs_per_trainer()) {
    comm->split_trainers(procs_per_trainer);
  }

  return procs_per_trainer;
}

/// Construct a trainer that contains a lbann comm object and threadpool
std::unique_ptr<trainer> construct_trainer(lbann_comm *comm,
                                           lbann_data::Trainer* pb_trainer,
                                           lbann_data::LbannPB &pb,
                                           options *opts) {
  try {
    if (pb_trainer->num_parallel_readers() > comm->get_procs_per_trainer()) {
      pb_trainer->set_num_parallel_readers(comm->get_procs_per_trainer());
    }

    // Adjust the number of parallel readers; this may be adjusted
    // after calling split_trainers()
    // set_num_parallel_readers(*comm, pb);

    // Check to see if the model wants to reduce the I/O parallelism
    bool serialized_io = false;
    if(pb_trainer->serialize_io()) {
      std::cout << "Trainer " << pb_trainer->name() << " serialized the I/O threads" << std::endl;
      serialized_io = true;
    }

    // Initalize a per-trainer I/O thread pool
    std::unique_ptr<thread_pool> io_thread_pool = construct_io_thread_pool(comm, opts, serialized_io);

    // Setup I/O threads
    auto io_threads_per_process = io_thread_pool->get_num_threads();
    auto io_threads_offset = io_thread_pool->get_threads_offset();

    // Set algorithmic blocksize in Hydrogen
    if (pb_trainer->hydrogen_block_size() > 0) {
      El::SetBlocksize(pb_trainer->hydrogen_block_size());
    }

    // Display how the OpenMP threads are provisioned
    // if (opts->has_string("print_affinity")) {
    //   display_omp_setup();
    // }

    // User feedback
    //    print_parameters(comm, pb);

    // Initalize trainer
    std::unique_ptr<trainer> trainer = proto::construct_trainer(comm, *pb_trainer);

    // If the checkpoint directory has been overridden reset it before
    // setting up the trainer
    if (opts && opts->has_string("ckpt_dir")) {
      for (auto&& c : trainer->get_callbacks()) {
        {
          auto* cb = dynamic_cast<callback::checkpoint*>(c);
          if(cb != nullptr) {
            cb->set_checkpoint_dir(opts->get_string("ckpt_dir"));
            if(comm->am_trainer_master()) {
              std::cout << "Setting the checkpoint directory to " << cb->get_checkpoint_dir() << std::endl;
            }
          }
        }
      }
    }
    if (opts && opts->has_string("restart_dir")) {
      for (auto&& c : trainer->get_callbacks()) {
        {
          auto* cb = dynamic_cast<callback::checkpoint*>(c);
          if(cb != nullptr) {
            cb->set_restart_dir(opts->get_string("restart_dir"));
            if(comm->am_trainer_master()) {
              std::cout << "Setting the restart directory to " << cb->get_restart_dir() << std::endl;
            }
          }
        }
      }
    }

    // Root of the random seed tree
    int root_random_seed = lbann_default_random_seed;

    // Change random seed if requested
    if (pb_trainer->random_seed() > 0) {
      root_random_seed = pb_trainer->random_seed();
    }

    // Random seed used for the general RNGs
    int random_seed = root_random_seed;
    // Random seed used for the RNG used to fetch data
    int data_seq_random_seed = root_random_seed;

    // Initialize models differently if needed.
#ifndef LBANN_DETERMINISTIC
    if (!pb_trainer->random_init_trainers_identically()) {
      random_seed = hash_combine(random_seed, comm->get_trainer_rank());
      // Also update the data sequence random seed
      data_seq_random_seed = random_seed;
    }

    // Under normal conditions, reinitialize the random number generator so
    // that regularization techniques (e.g. dropout) generate unique patterns
    // on different ranks.
    // At this point the data sequence random seed is no longer updated
    random_seed = hash_combine(random_seed, comm->get_rank_in_world());
#else
    if(comm->am_world_master()) {
      std::cout <<
        "--------------------------------------------------------------------------------------------------------------------\n"
        "ALERT: executing with LBANN_DETERMINISTIC flag to minimize reduce numerical variance -- performance will be degraded\n"
        "--------------------------------------------------------------------------------------------------------------------\n";
    }
    if (!pb_trainer->random_init_trainers_identically()) {
      if(comm->am_trainer_master()) {
        std::cout << "WARNING: forcing 'random_init_trainers_identically' " <<
          "due to sequential consistency" << std::endl;
      }
    }
#endif

    // Initialize the general RNGs and the data sequence RNGs
    init_random(random_seed, io_threads_per_process);
    init_data_seq_random(data_seq_random_seed);
    init_ltfb_random(root_random_seed);
    trainer->set_random_seeds(root_random_seed, random_seed, data_seq_random_seed);

    // Collect everyone's random seeds
    std::vector<int> root_random_seeds(comm->get_procs_in_world());
    comm->world_all_gather(root_random_seed, root_random_seeds);
    std::vector<int> random_seeds(comm->get_procs_in_world());
    comm->world_all_gather(random_seed, random_seeds);
    std::vector<int> data_seq_random_seeds(comm->get_procs_in_world());
    comm->world_all_gather(data_seq_random_seed, data_seq_random_seeds);

    // Update the sample lists to accomodate multi-trainer / multi-model specification
    customize_data_readers_sample_list(*comm, pb);

    // Initialize data readers
    //@todo: code not in place for correctly handling image preprocessing
    std::map<execution_mode, generic_data_reader *> data_readers;
    bool is_shared_training_data_reader = pb_trainer->shareable_training_data_reader();
    bool is_shared_testing_data_reader = pb_trainer->shareable_testing_data_reader();
    if (opts->has_string("share_testing_data_readers")) {
      is_shared_testing_data_reader = opts->get_bool("share_testing_data_readers");
    }
    init_data_readers(comm, pb, data_readers, is_shared_training_data_reader, is_shared_testing_data_reader);

    trainer->setup(std::move(io_thread_pool), data_readers);

    if(opts->get_bool("disable_background_io_activity")) {
      trainer->allow_background_io_activity(false);
    }


    // Report useful information
    if (comm->am_world_master()) {
      print_lbann_configuration(comm,
                                io_threads_per_process,
                                io_threads_offset);
      std::cout << "\n"
                << trainer->get_description()
                << std::endl;

      // User feedback
      print_parameters(*comm, pb, root_random_seeds, random_seeds, data_seq_random_seeds);
    }

    return trainer;

  } catch (lbann_exception& e) {
    El::mpi::Abort(El::mpi::COMM_WORLD, 1);
  } catch (std::exception& e) {
    El::ReportException(e);  // Elemental exceptions
  }
  return nullptr;
}

/// Setup I/O thread pool that is shared across all models
  std::unique_ptr<thread_pool> construct_io_thread_pool(lbann_comm *comm, options *opts, bool serialized_io) {
  int max_io_threads = num_free_cores_per_process(comm);
  // Allow the trainer to override the command-line option or environment variable
  if(serialized_io) {
    max_io_threads = 1;
  }

  auto& arg_parser = global_argument_parser();
  int req_io_threads = arg_parser.get<int>(NUM_IO_THREADS);
  int num_io_threads = std::max(std::min(max_io_threads, req_io_threads), 1);

  auto io_threads_offset = free_core_offset(comm);

  if(comm->am_world_master()) {
    std::cout << "\tNum. I/O Threads: " << num_io_threads <<
      " (Limited to # Unused Compute Cores or 1) at offset "
      << io_threads_offset << std::endl;
  }

  auto io_thread_pool = make_unique<thread_pool>();
  io_thread_pool->launch_pinned_threads(num_io_threads, io_threads_offset);

  return io_thread_pool;
}

std::unique_ptr<model> build_model_from_prototext(
  int argc, char **argv,
  const lbann_data::Trainer* pb_trainer,
  lbann_data::LbannPB &pb,
  lbann_comm *comm,
  options *opts,
  thread_pool& io_thread_pool,
  std::vector<std::shared_ptr<callback_base>>& shared_callbacks,
  int training_dr_linearized_data_size) {

  bool master = comm->am_world_master();
  if (master) {
    std::cerr << "starting build_model_from_prototext" << std::endl;
  }

  std::ostringstream err;

  // Save info to file; this includes the complete prototext (with any over-rides
  // from the cmd line) and various other info
  save_session(*comm, argc, argv, pb);

  // Display how the OpenMP threads are provisioned
  if (opts->has_string("print_affinity")) {
    display_omp_setup();
  }

  // Initalize model
  std::unique_ptr<model> ret_model = proto::construct_model(comm,
                                                            training_dr_linearized_data_size,
                                                            pb.optimizer(),
                                                            pb.trainer(),
                                                            pb.model());

  // Add the trainer's callbacks to the model
  for (auto&& c : shared_callbacks) {
    ret_model->add_callback(c);
  }

  // If the checkpoint directory has been overridden reset it before
  // setting up the model
  if (opts && opts->has_string("ckpt_dir")) {
    for (auto&& c : ret_model->get_callbacks()) {
      {
        auto* cb = dynamic_cast<callback::dump_weights*>(c);
        if(cb != nullptr) {
          cb->set_target_dir(opts->get_string("ckpt_dir"));
          if(comm->am_trainer_master()) {
            std::cout << "Setting the dump weights directory to " << cb->get_target_dir() << std::endl;
          }
        }
      }
      {
        auto* cb = dynamic_cast<callback::save_model*>(c);
        if(cb != nullptr) {
          cb->set_target_dir(opts->get_string("ckpt_dir"));
          if(comm->am_trainer_master()) {
            std::cout << "Setting the dump weights directory to " << cb->get_target_dir() << std::endl;
          }
        }
      }
    }
  }

  if (opts && opts->has_string("load_model_weights_dir")) {
    callback::load_model* cb = nullptr;
    for (auto&& c : ret_model->get_callbacks()) {
      cb = dynamic_cast<callback::load_model*>(c);
      if(cb != nullptr) {
        break;
      }
    }

    std::string active_load_model_dir;
    std::string load_model_dir = opts->get_string("load_model_weights_dir");
    if(opts->get_bool("load_model_weights_dir_is_complete")) {
      active_load_model_dir = load_model_dir;
    }else {
      size_t epochLast = std::numeric_limits<size_t>::max();;
      size_t stepLast = std::numeric_limits<size_t>::max();;
      execution_mode mode = execution_mode::invalid;
      visitor_hook hook = visitor_hook::invalid;
      active_load_model_dir = callback::get_last_shared_checkpoint_filename("sgd", load_model_dir);

      // get last epoch and step saved.
      int success = callback::read_latest(active_load_model_dir, &hook, &mode, &epochLast, &stepLast);
      if(!success) {
        LBANN_ERROR("Unable to find the latest checkpoint ", active_load_model_dir);
        return nullptr;
      }
      active_load_model_dir = callback::get_shared_checkpoint_dirname("sgd", load_model_dir, hook, mode, epochLast, stepLast) + ret_model->get_name() + '/';
    }

    if(cb == nullptr) {
      std::vector<std::string> dirs = {active_load_model_dir};
      std::unique_ptr<callback::load_model> load_model_cb =
        make_unique<callback::load_model>(dirs);
      cb = load_model_cb.get();
      ret_model->add_callback(std::move(load_model_cb));
#ifdef LBANN_DEBUG
      if(comm->am_trainer_master()) {
        LBANN_WARNING("command line flag --load_model_dir was provided but there was no explicit load_model callback, adding one automagically!");
      }
#endif
    }else {
      cb->add_dir(opts->get_string("load_model_weights_dir"));
    }
  }

  // restart model from checkpoint if we have one
  //@todo
  //model->restartShared();

  return ret_model;
}

void print_lbann_configuration(lbann_comm *comm, int io_threads_per_process, int io_threads_offset) {
  // Report hardware settings
  std::cout << "Hardware properties (for master process)" << std::endl
            << "  Processes on node          : " << comm->get_procs_per_node() << std::endl
            << "  Total number of processes  : " << comm->get_procs_in_world() << std::endl
            << "  OpenMP threads per process : " << omp_get_max_threads() << std::endl
            << "  I/O threads per process (+offset) : " << io_threads_per_process
            << " (+" << io_threads_offset << ")" << std::endl;
#ifdef HYDROGEN_HAVE_GPU
  std::cout << "  GPUs on node               : " << hydrogen::gpu::DeviceCount() << std::endl;
#endif // HYDROGEN_HAVE_GPU
  std::cout << std::endl;

  // Report build settings
  std::cout << "Running: LLNL LBANN version: "
            << LBANN_MAKE_STR(LBANN_VERSION)
            << " (" << LBANN_MAKE_STR(LBANN_GIT_VERSION) << ")"
            << std::endl;
#ifdef HYDROGEN_VERSION
  std::cout << "         LLNL Hydrogen version: "
            << HYDROGEN_VERSION
            << " (" << HYDROGEN_GIT_VERSION << ")"
            << std::endl << std::endl;
#endif
  std::cout << "Build settings" << std::endl;
  std::cout << "  Type     : ";
#ifdef LBANN_DEBUG
  std::cout << "Debug" << std::endl;
#else
  std::cout << "Release" << std::endl;
#endif // LBANN_DEBUG
  std::cout << "  Aluminum : ";
#ifdef LBANN_HAS_ALUMINUM
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_ALUMINUM
  std::cout << "  GPU     : ";
#ifdef LBANN_HAS_GPU
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_GPU
  std::cout << "  cuDNN    : ";
#ifdef LBANN_HAS_CUDNN
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_CUDNN
  std::cout << "  CUB      : ";
#ifdef HYDROGEN_HAVE_CUB
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // HYDROGEN_HAVE_CUB
  const auto* env = std::getenv("MV2_USE_CUDA");
  std::cout << "  MV2_USE_CUDA : " << (env != nullptr ? env : "") << std::endl;
  std::cout << std::endl;

#ifdef LBANN_HAS_ALUMINUM
  std::cout << "Aluminum Features:" << std::endl;
  std::cout << "  NCCL : ";
#ifdef AL_HAS_NCCL
  std::cout << "enabled" << std::endl;
#else
  std::cout << "disabled" << std::endl;
#endif // AL_HAS_NCCL
  std::cout << std::endl;
#endif // LBANN_HAS_ALUMINUM

  // Report model settings
  const auto& grid = comm->get_trainer_grid();
  std::cout << "Trainer settings" << std::endl
            << "  Trainers              : " << comm->get_num_trainers() << std::endl
            << "  Processes per trainer : " << comm->get_procs_per_trainer() << std::endl
            << "  Grid dimensions       : " << grid.Height() << " x " << grid.Width() << std::endl;
  std::cout << std::endl;
}

} // namespace lbann
