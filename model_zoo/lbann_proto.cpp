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
#include "lbann/proto/lbann_proto_common.hpp"
//#include "lbann/callbacks/lbann_callback_ltfb.hpp"

using namespace lbann;

int main(int argc, char *argv[]) {
#if 0
  lbann_comm *comm = initialize(argc, argv, 42);

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {
    options *opts = options::get();
    std::stringstream err;


    //run LTFB?
    bool ltfb = opts->has_int("ltfb") and opts->get_int("ltfb");

    //get input prototext filenames;
    if (not opts->has_string("proto") and (comm->am_world_master()) {
      err << __FILE__ << " " << __LINE__
          << " :: you must pass the option: --proto=<string>\n"
          << "and, unless you're running from a prototext file saved from\n"
          << "a previous run, you probably also need to specify\n"
          << "--proto_reader=<string> and --proto_optimizer=<string>\n";
      throw lbann_exception(err.str());
    }
   
    string prototext_model_fn = opts->get_string("proto");
    lbann_data::LbannPB pb;
    readPrototextFile(prototext_model_fn.c_str(), pb);

    if (opts->has_string("proto_reader")) {
      lbann_data::LbannPB pb_reader;
      readPrototextFile(opts->get_string("proto_reader").c_str(), pb_reader);
      pb.MergeFrom(pb_reader);
    }

    if (opts->has_string("proto_optimizer")) {
      string prototext_opt_fn;
      lbann_data::LbannPB pb_optimizer;
      readPrototextFile(opts->get_string("proto_optimizer").c_str(), pb_optimizer);
      pb.MergeFrom(pb_optimizer);
    }

    lbann_data::Model *pb_model = pb.mutable_model();

    set_num_parallel_readers(comm, pb);
    get_cmdline_overrides(comm, pb);

#if 0
void set_num_parallel_readers(lbann::lbann_comm *comm, lbann_data::LbannPB& p) {
    int parallel_io = pb_model->num_parallel_readers();
    if (parallel_io == 0) {
      if (comm->am_world_master()) {
        cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
             " (Limited to # Processes)" << endl;
      }
      parallel_io = comm->get_procs_per_model();
      pb_model->set_num_parallel_readers(parallel_io); //adjust the prototext
    } else {
      if (comm->am_world_master()) {
        cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
      }
    }
}

void get_cmd_overrides(lbann::lbann_comm *comm, lbann_data::LbannPB& p) {
    int mini_batch_size = Input("--mb-size", "mini_batch_size", 0);
    int num_epochs = Input("--num-epochs", "num epochs", 0);

    if (mini_batch_size != 0) {
      pb_model->set_mini_batch_size(mini_batch_size);
    }
    if (num_epochs != 0) {
      pb_model->set_num_epochs(num_epochs);
    }
}
#endif

    // Set algorithmic blocksize
    if (pb_model->block_size() == 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: model does not provide a valid block size: " << pb_model->block_size();
      throw lbann_exception(err.str());
    }
    SetBlocksize(pb_model->block_size());

    // Set up the communicator and get the grid.
    comm->split_models(pb_model->procs_per_model());
    if (comm->am_world_master()) cout << "procs_per_model: " << pb_model->procs_per_model() << endl;
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      cout << "Number of models: " << comm->get_num_models() << endl;
      cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << endl;
    }


    ///////////////////////////////////////////////////////////////////
    // initialize data readers
    //@todo: code not in place for correctly handling image preprocessing
    ///////////////////////////////////////////////////////////////////
    std::map<execution_mode, generic_data_reader *> data_readers;
    init_data_readers(comm->am_world_master(), pb, data_readers, pb_model->mini_batch_size());
    if (comm->am_world_master()) {
      for (auto it : data_readers) {
        cerr << "data reader; role: " << it.second->get_role()
             << " num data: " << it.second->getNumData() << endl;
      }
    }

    //user feedback
    if (comm->am_world_master()) {
      cout << "\nParameter settings:" << endl;
      cout << "\tMini-batch size: " << pb_model->mini_batch_size() << endl;
      const lbann_data::Optimizer optimizer = pb.optimizer();
      cout << "\tLearning rate: " <<  optimizer.learn_rate() << endl;
      cout << "\tEpoch count: " << pb_model->num_epochs() << endl << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize model
    // @todo: not all callbacks code is in place
    ///////////////////////////////////////////////////////////////////
    optimizer_factory *optimizer_fac = init_optimizer_factory(comm, pb);
    cudnn::cudnn_manager *cudnn = NULL;
#if __LIB_CUDNN
    if (pb_model->use_cudnn()) {
      if (comm->am_world_master()) {
        cerr << "USING cudnn\n";
      }
      cudnn = new cudnn::cudnn_manager(comm, pb_model->num_gpus());
    } else {
      if (comm->am_world_master()) {
        cerr << "code was compiled with __LIB_CUDNN, but we are NOT USING cudnn\n";
      }
    }
#else
    if (comm->am_world_master()) {
      cerr << "code was NOT compiled with __LIB_CUDNN\n";
    }
#endif
    sequential_model *model = init_model(comm, optimizer_fac, pb);
    std::unordered_map<uint,uint> layer_mapping;
    add_layers(model, data_readers, cudnn, pb, layer_mapping);
    init_callbacks(comm, model, data_readers, pb, layer_mapping);
    model->setup();

    // Optionally run ltfb
    sequential_model *model_2;
    std::map<execution_mode, generic_data_reader *> data_readers_2;
    if (ltfb) {
      if (comm->am_world_master()) {
        cerr << endl << "running ltfb\n\n";
        throw lbann_exception("ltfb is not ready yet; coming soon!");
      }
      init_data_readers(comm->am_world_master(), pb, data_readers_2, pb_model->mini_batch_size());
      optimizer_factory *optimizer_fac_2 = init_optimizer_factory(comm, pb);
      model_2 = init_model(comm, optimizer_fac_2, pb);
      model_2->setup();
      //lbann_callback_ltfb ltfb(45, model_2);
      //model->add_callback(&ltfb);
    }

    // restart model from checkpoint if we have one
    //@todo
    //model->restartShared();

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////
    if (pb_model->data_layout() == "model_parallel") {
      if (comm->am_world_master()) {
        cout << "MODEL_PARALLEL, so calling: init_random(comm->get_rank_in_world() + 1)\n";
      }
      init_random(comm->get_rank_in_world() + 1);
    }
    while (model->get_cur_epoch() < pb_model->num_epochs()) {
      model->train(1, true);
      model->evaluate(execution_mode::testing);
    }

    // @todo: figure out and implement coherent strategy
    // for freeing dynamically allocated memory
  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (exception& e) {
    ReportException(e);  /// Elemental exceptions
  }

  // free all resources by El and MPI
  finalize(comm);
#endif
  return 0;
}
