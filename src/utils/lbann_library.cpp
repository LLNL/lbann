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
#include "lbann/callbacks/callback_checkpoint.hpp"

namespace lbann {

/// Setup I/O thread pool that is shared across all models
std::unique_ptr<thread_pool> construct_io_thread_pool(lbann_comm *comm) {
  int num_io_threads = num_free_cores_per_process(comm);

  options *opts = options::get();
  if(opts->has_int("num_io_threads")) {
    int requested_io_threads = opts->get_int("num_io_threads");
    if(requested_io_threads > 0 && requested_io_threads < num_io_threads) {
      num_io_threads = requested_io_threads;
    }
  }

  auto io_threads_offset = free_core_offset(comm);

  if(comm->am_world_master()) {
    std::cout << "\tNum. I/O Threads: " << num_io_threads <<
      " (Limited to # Unused Compute Cores or 1)" << std::endl;
  }

  auto io_thread_pool = make_unique<thread_pool>();
  io_thread_pool->launch_pinned_threads(num_io_threads, io_threads_offset);

  return io_thread_pool;
}

std::unique_ptr<model> build_model_from_prototext(
  int argc, char **argv,
  lbann_data::LbannPB &pb,
  lbann_comm *comm,
  std::shared_ptr<thread_pool> io_thread_pool,
  bool first_model) {

  int random_seed = lbann_default_random_seed;
  bool master = comm->am_world_master();
  if (master) {
    std::cerr << "starting build_model_from_prototext" << std::endl;
  }

  std::ostringstream err;
  options *opts = options::get();

  // Optionally over-ride some values in prototext
  get_cmdline_overrides(*comm, pb);

  lbann_data::Model *pb_model = pb.mutable_model();

  // Adjust the number of parallel readers; this may be adjusted
  // after calling split_trainers()
  set_num_parallel_readers(*comm, pb);

  // Check to see if the model wants to reduce the I/O parallelism
  if(pb_model->serialize_io() && io_thread_pool->get_num_threads() != 1) {
    if(master) {
      std::cout << "Model " << pb_model->name() << " serialized the I/O threads" << std::endl;
    }
    io_thread_pool->relaunch_pinned_threads(1);
  }

  // Setup I/O threads
  auto io_threads_per_process = io_thread_pool->get_num_threads();
  auto io_threads_offset = io_thread_pool->get_threads_offset();

  // Set algorithmic blocksize
  if (pb_model->block_size() == 0 and master) {
    err << "model does not provide a valid block size (" << pb_model->block_size() << ")";
    LBANN_ERROR(err.str());
  }
  El::SetBlocksize(pb_model->block_size());

  // Change random seed if needed.
  if (pb_model->random_seed() > 0) {
    random_seed = pb_model->random_seed();
    // Reseed here so that setup is done with this new seed.
    init_random(random_seed);
    init_data_seq_random(random_seed);
  }
  // Set up the communicator and get the grid based on the first model's spec.
  // We do not currently support splitting different models in different ways,
  // as this implies different grids.
  int procs_per_trainer = pb_model->procs_per_trainer();
  if (procs_per_trainer == 0) {
    procs_per_trainer = comm->get_procs_in_world();
  }
  if (first_model) {
    comm->split_trainers(procs_per_trainer);
    if (pb_model->num_parallel_readers() > procs_per_trainer) {
      pb_model->set_num_parallel_readers(procs_per_trainer);
    }
  } else if (procs_per_trainer != comm->get_procs_per_trainer()) {
    LBANN_ERROR("Model prototexts requesting different procs per model is not supported");
  }

  // Initialize models differently if needed.
#ifndef LBANN_DETERMINISTIC
  if (pb_model->random_init_models_differently()) {
    random_seed = random_seed + comm->get_trainer_rank();
    // Reseed here so that setup is done with this new seed.
    init_random(random_seed);
    init_data_seq_random(random_seed);
  }
#else
  if (pb_model->random_init_models_differently()) {
    if (master) {
      std::cout << "WARNING: Ignoring random_init_models_differently " <<
        "due to sequential consistency" << std::endl;
    }
  }
#endif

  // Save info to file; this includes the complete prototext (with any over-rides
  // from the cmd line) and various other info
  save_session(*comm, argc, argv, pb);

  // Report useful information
  if (master) {
    print_lbann_configuration(pb_model, comm, io_threads_per_process, io_threads_offset);
  }

  // Display how the OpenMP threads are provisioned
  if (opts->has_string("print_affinity")) {
    display_omp_setup();
  }

  // Update the index lists to accomodate multi-trainer / multi-model specification
  customize_data_readers_index_list(*comm, pb);

  // Initialize data readers
  //@todo: code not in place for correctly handling image preprocessing
  std::map<execution_mode, generic_data_reader *> data_readers;
  bool is_shared_training_data_reader = pb_model->shareable_training_data_reader();
  bool is_shared_testing_data_reader = pb_model->shareable_testing_data_reader();
  if (opts->has_string("share_testing_data_readers")) {
    is_shared_testing_data_reader = opts->get_bool("share_testing_data_readers");
  }
  init_data_readers(comm, pb, data_readers, is_shared_training_data_reader, is_shared_testing_data_reader);

  // hack to prevent all data readers from loading identical data; instead,
  // share a single copy. See data_reader_jag_conduit_hdf5 for example
  if (first_model) {
    if (opts->has_string("share_data_reader_data")) {
      for (auto&& t : data_readers) {
        opts->set_ptr((void*)t.second);
      }
    }
  }

  // User feedback
  print_parameters(*comm, pb);

  // Initalize model
  std::unique_ptr<model> ret_model{
    proto::construct_model(comm,
                           data_readers,
                           pb.optimizer(),
                           pb.model())
  };
  ret_model->setup(std::move(io_thread_pool));

  if(opts->get_bool("disable_background_io_activity")) {
    ret_model->allow_background_io_activity(false);
  }

  if (opts->get_bool("use_data_store") || opts->get_bool("preload_data_store")) {
    if (master) {
      std::cout << "\nUSING DATA STORE!\n\n";
    }
    for (auto&& r : data_readers) {
      if (!r.second) continue;
      r.second->setup_data_store(pb_model->mini_batch_size());
    }
  }

  // restart model from checkpoint if we have one
  //@todo
  //model->restartShared();

  if (comm->am_world_master()) {
    std::cout << "\n"
              << ret_model->get_description()
              << "Callbacks:" << std::endl;
    for (lbann_callback *cb : ret_model->get_callbacks()) {
      std::cout << cb->name() << std::endl;
    }
  }

#ifndef LBANN_DETERMINISTIC
  // Under normal conditions, reinitialize the random number generator so
  // that regularization techniques (e.g. dropout) generate unique patterns
  // on different ranks.
  init_random(random_seed + comm->get_rank_in_world());
#else
  if(comm->am_world_master()) {
    std::cout <<
      "--------------------------------------------------------------------------------\n"
      "ALERT: executing in sequentially consistent mode -- performance will suffer\n"
      "--------------------------------------------------------------------------------\n";
  }
#endif
  return ret_model;
}

void print_lbann_configuration(lbann_data::Model *pb_model, lbann_comm *comm, int io_threads_per_process, int io_threads_offset) {
  // Report hardware settings
  std::cout << "Hardware properties (for master process)" << std::endl
            << "  Processes on node          : " << comm->get_procs_per_node() << std::endl
            << "  Total number of processes  : " << comm->get_procs_in_world() << std::endl
            << "  OpenMP threads per process : " << omp_get_max_threads() << std::endl
            << "  I/O threads per process (+offset) : " << io_threads_per_process
            << " (+" << io_threads_offset << ")" << std::endl;
#ifdef HYDROGEN_HAVE_CUDA
  std::cout << "  GPUs on node               : " << El::GPUManager::NumDevices() << std::endl;
#endif // HYDROGEN_HAVE_CUDA
  std::cout << std::endl;

  // Report build settings
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
  std::cout << "  CUDA     : ";
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
  std::cout << std::endl;

  // Report device settings
  std::cout << "GPU settings" << std::endl;
  bool disable_cuda = pb_model->disable_cuda();
#ifndef LBANN_HAS_GPU
  disable_cuda = true;
#endif // LBANN_HAS_GPU
  std::cout << "  CUDA         : "
            << (disable_cuda ? "disabled" : "enabled") << std::endl;
  std::cout << "  cuDNN        : ";
#ifdef LBANN_HAS_CUDNN
  std::cout << (disable_cuda ? "disabled" : "enabled") << std::endl;
#else
  std::cout << "disabled" << std::endl;
#endif // LBANN_HAS_CUDNN
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
  int procs_per_trainer = pb_model->procs_per_trainer();
  std::cout << "Model settings" << std::endl
            << "  Models                : " << comm->get_num_trainers() << std::endl
            << "  Processes per trainer : " << procs_per_trainer << std::endl
            << "  Grid dimensions       : " << grid.Height() << " x " << grid.Width() << std::endl;
  std::cout << std::endl;
}

} // namespace lbann
