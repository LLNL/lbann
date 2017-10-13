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

using namespace lbann;

const int lbann_random_seed = 42;

int main(int argc, char *argv[]) {
  lbann_comm *comm = initialize(argc, argv, lbann_random_seed);
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

    // Run LTFB?
    //bool ltfb = opts->has_int("ltfb") and opts->get_int("ltfb");

    // Get input prototext filename(s)
    if (not (opts->has_string("loadme") or opts->has_string("model"))) {
      if (master) {  
        err << __FILE__ << " " << __LINE__
            << 
            " :: you must either pass the option: --loadme=<string> (if the file\n"
            "contains a specification for the model, readers, and optimizer\n"
             "or --model=<string> --reader=<string> --optimizer=<string>\n";
      throw lbann_exception(err.str());
    }
    }
   
    lbann_data::LbannPB pb;
    string prototext_model_fn;
    if (opts->has_string("model")) {
      prototext_model_fn = opts->get_string("model");
    } else if (opts->has_string("loadme")) {
      prototext_model_fn = opts->get_string("loadme");
    } 
    read_prototext_file(prototext_model_fn.c_str(), pb, master);

    if (opts->has_string("reader")) {
      lbann_data::LbannPB pb_reader;
      read_prototext_file(opts->get_string("reader").c_str(), pb_reader, master);
      pb.MergeFrom(pb_reader);
    }

    if (opts->has_string("optimizer")) {
      string prototext_opt_fn;
      lbann_data::LbannPB pb_optimizer;
      read_prototext_file(opts->get_string("optimizer").c_str(), pb_optimizer, master);
      pb.MergeFrom(pb_optimizer);
    }

    lbann_data::Model *pb_model = pb.mutable_model();

    // Optionally over-ride some values in prototext
    get_cmdline_overrides(comm, pb);

    // Adjust the number of parallel readers; this may be adjusted
    // after calling split_models()
    set_num_parallel_readers(comm, pb);

    // Save info to file; this includes the complete prototext (with any over-rides

    // Set algorithmic blocksize
    if (pb_model->block_size() == 0 and master) {
      err << __FILE__ << " " << __LINE__ << " :: model does not provide a valid block size: " << pb_model->block_size();
      throw lbann_exception(err.str());
    }
    SetBlocksize(pb_model->block_size());

    // Set up the communicator and get the grid.
    int procs_per_model = pb_model->procs_per_model();
    if (procs_per_model == 0) {
      procs_per_model = comm->get_procs_in_world();
    }
    comm->split_models(procs_per_model);
    if (master) cout << "  procs_per_model: " << procs_per_model << endl;
    if (pb_model->num_parallel_readers() > procs_per_model) {
      pb_model->set_num_parallel_readers(procs_per_model);
    }

    Grid& grid = comm->get_model_grid();
    if (master) {
      cout << "  Number of models: " << comm->get_num_models() << endl;
      cout << "  Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << endl;
    }

    // from the cmd line) and various other info
    save_session(comm, argc, argv, pb);

    // Initialize data readers
    //@todo: code not in place for correctly handling image preprocessing
    std::map<execution_mode, generic_data_reader *> data_readers;
    init_data_readers(master, pb, data_readers, pb_model->mini_batch_size());

    // Check for cudnn, with user feedback
    cudnn::cudnn_manager *cudnn = NULL;
#if __LIB_CUDNN
    if (pb_model->use_cudnn()) {
      if (master) {
        cerr << "code was compiled with __LIB_CUDNN, and we are using cudnn\n";
      }
      cudnn = new cudnn::cudnn_manager(comm, pb_model->num_gpus());
    } else {
      if (master) {
        cerr << "code was compiled with __LIB_CUDNN, but we are NOT USING cudnn\n";
      }
    }
#else
    if (master) {
      cerr << "code was NOT compiled with __LIB_CUDNN\n";
    }
#endif

    // Construct optimizer
    optimizer_factory *optimizer_fac = init_optimizer_factory(comm, cudnn, pb);

    // User feedback
    print_parameters(comm, pb);

    // Initalize model
    // @todo: not all callbacks code is in place
    model *model = init_model(comm, optimizer_fac, pb);
    add_layers(model, data_readers, cudnn, pb);
    init_callbacks(comm, model, data_readers, pb);
    model->setup();

    // restart model from checkpoint if we have one
    //@todo
    //model->restartShared();

    if (comm->am_world_master()) {
      optimizer *o = optimizer_fac->create_optimizer();
      cout << "\nOptimizer:\n" << o->get_description() << endl << endl;
      delete o;
      std::vector<Layer *>& layers = model->get_layers();
      for (size_t h=0; h<layers.size(); h++) {
        std::cout << h << " " << layers[h]->get_description() << endl;
      }
    }

    if (not opts->has_string("exit_after_setup")) {

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////

#ifndef LBANN_SEQUENTIAL_CONSISTENCY
    // Under normal conditions, reinitialize the random number generator so
    // that regularization techniques (e.g. dropout) generate unique patterns
    // on different ranks.
    init_random(lbann_random_seed + comm->get_rank_in_world());
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

    } 

    else {
      if (comm->am_world_master()) {
        std::cout << 
          "--------------------------------------------------------------------------------\n"
          "ALERT: model has been setup; we are now exiting due to command\n"
          "       line option: --exit_after_setup\n"
          "--------------------------------------------------------------------------------\n";
      }
    }

    // @todo: figure out and implement coherent strategy
    // for freeing dynamically allocated memory
    delete model;
    delete optimizer_fac;

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  // free all resources by El and MPI
  finalize(comm);
  return 0;
}
