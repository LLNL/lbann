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
#include "lbann/utils/stack_profiler.hpp"
#include "lbann/data_store/generic_data_store.hpp"


using namespace lbann;

const int lbann_default_random_seed = 42;


int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  if (master) {
    std::cout << "\n\n==============================================================\n"
              << "STARTING lbann with this command line:\n";
    for (int j=0; j<argc; j++) {
      std::cout << argv[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

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



    //this must be called after call to opts->init();
    //must also specify "--catch-signals" on cmd line
    stack_trace::register_handler();

    //to activate, must specify --st_on on cmd line
    stack_profiler::get()->activate(comm->get_rank_in_world());

    std::vector<lbann_data::LbannPB *> pbs;
    protobuf_utils::load_prototext(master, argc, argv, pbs);
    lbann_data::LbannPB pb = *(pbs[0]);


    lbann_data::Model *pb_model = pb.mutable_model();

    // Optionally over-ride some values in prototext
    get_cmdline_overrides(comm, pb);

    // Adjust the number of parallel readers; this may be adjusted
    // after calling split_models()
    set_num_parallel_readers(comm, pb);

    // Set algorithmic blocksize
    if (pb_model->block_size() == 0 and master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: model does not provide a valid block size: " << pb_model->block_size();
      throw lbann_exception(err.str());
    }
    El::SetBlocksize(pb_model->block_size());

    // Change random seed if needed.
    if (pb_model->random_seed() > 0) {
      random_seed = pb_model->random_seed();
      // Reseed here so that setup is done with this new seed.
      init_random(random_seed);
      init_data_seq_random(random_seed);
    }
    // Initialize models differently if needed.
#ifndef LBANN_SEQUENTIAL_CONSISTENCY
    if (pb_model->random_init_models_differently()) {
      random_seed = random_seed + comm->get_model_rank();
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

    // Set up the communicator and get the grid.
    int procs_per_model = pb_model->procs_per_model();
    if (procs_per_model == 0) {
      procs_per_model = comm->get_procs_in_world();
    }
    comm->split_models(procs_per_model);
    if (pb_model->num_parallel_readers() > procs_per_model) {
      pb_model->set_num_parallel_readers(procs_per_model);
    }

    if (master) {
      std::cout << "Model settings" << std::endl
                << "  Models              : " << comm->get_num_models() << std::endl
                << "  Processes per model : " << procs_per_model << std::endl
                << "  Grid dimensions     : " << comm->get_model_grid().Height() << " x " << comm->get_model_grid().Width() << std::endl;
      std::cout << std::endl;
    }

    // Save info to file; this includes the complete prototext (with any over-rides
    // from the cmd line) and various other info
    save_session(comm, argc, argv, pb);

    // Check for cudnn, with user feedback
    cudnn::cudnn_manager *cudnn = nullptr;
#ifdef LBANN_HAS_CUDNN
    const size_t workspace_size = 1 << 9; // 1 GB
    if (! pb_model->disable_cuda()) {
      if (master) {
        std::cerr << "code was compiled with LBANN_HAS_CUDNN, and we are using cudnn\n";
      }
      cudnn = new cudnn::cudnn_manager(workspace_size);
    } else {
      if (master) {
        std::cerr << "code was compiled with LBANN_HAS_CUDNN, but we are NOT USING cudnn\n";
      }
    }
#else
    if (master) {
      std::cerr << "code was NOT compiled with LBANN_HAS_CUDNN\n";
    }
#endif

    if (master) {
      std::cout << "Hardware settings (for master process)" << std::endl
                << "  Processes on node            : " << comm->get_procs_per_node() << std::endl
                << "  OpenMP threads per process   : " << omp_get_max_threads() << std::endl;
      #ifdef LBANN_HAS_CUDNN
      if (cudnn != nullptr) {
        std::cout << "  GPUs on node                 : " << El::GPUManager::NumDevices() << std::endl;
      }
      #endif // LBANN_HAS_CUDNN
      std::cout << std::endl;
    }

    // Display how the OpenMP threads are provisioned
    if (opts->has_string("print_affinity")) {
      display_omp_setup();
    }

    // Initialize data readers
    //@todo: code not in place for correctly handling image preprocessing
    std::map<execution_mode, generic_data_reader *> data_readers;
    init_data_readers(comm, pb, data_readers);

    // User feedback
    print_parameters(comm, pb);

    // Initalize model
    auto&& model = proto::construct_model(comm,
                                          cudnn,
                                          data_readers,
                                          pb.optimizer(),
                                          pb.model());
    model->setup();

    //under development; experimental
    if (opts->has_bool("use_data_store") && opts->get_bool("use_data_store")) {
      if (master) {
        std::cerr << "\nUSING DATA STORE!\n\n";
      }
      for (auto r : data_readers) {
        r.second->setup_data_store(model);
      }
    }

    if (opts->has_string("create_tarball")) {
      finalize(comm);
      return 0;
    }

    // restart model from checkpoint if we have one
    //@todo

    if (comm->am_world_master()) {
      std::cout << std::endl;
      std::cout << "Callbacks:" << std::endl;
      for (lbann_callback *cb : model->get_callbacks()) {
        std::cout << cb->name() << std::endl;
      }
      std::cout << std::endl;
      const std::vector<Layer *>& layers = model->get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << std::endl;
      }
    }

    if (! (opts->has_bool("exit_after_setup") && opts->get_bool("exit_after_setup"))) {

#ifndef LBANN_SEQUENTIAL_CONSISTENCY
      // Under normal conditions, reinitialize the random number generator so
      // that regularization techniques (e.g. dropout) generate unique patterns
      // on different ranks.
      // Do not do this if current epoch/iter is not 0. 
      // Signifies a restart has occured and rng state has been loaded in. 
      if(model->get_cur_epoch() == 0 && model->get_cur_step() == 0){
        init_random(random_seed + comm->get_rank_in_world());
      }
#else
      if(comm->am_world_master()) {
        std::cout <<
          "--------------------------------------------------------------------------------\n"
          "ALERT: executing in sequentially consistent mode -- performance will suffer\n"
          "--------------------------------------------------------------------------------\n";
      }
#endif

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

    // @todo: figure out and implement coherent strategy
    // for freeing dynamically allocated memory
    delete model;

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (std::exception& e) {
    El::ReportException(e);  // Elemental exceptions
  }

  // free all resources by El and MPI
  finalize(comm);
  return 0;
}
