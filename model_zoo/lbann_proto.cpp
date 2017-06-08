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

using namespace std;
using namespace lbann;
using namespace El;

int main(int argc, char* argv[])
{
  Initialize(argc, argv);
  lbann_comm* comm = NULL;

  try {

    //get input prototext filenames
    string prototext_model_fn = Input("--prototext_fn", "prototext model filename", "none");
    string prototext_dr_fn = Input("--prototext_dr_fn", "prototext data reader filename", "none");

    //Get optional over-rides for various fields in the prototext.
    //If you add anything here, also change the section below:
    //  "adjust the prototext with any over-rides"
    int mini_batch_size = Input("--mb-size", "mini_batch_size", 0);
    int num_epochs = Input("--num-epochs", "num epochs", 0);

    ProcessInput();
    PrintInputReport();

    //error check the command line
    if (prototext_dr_fn == "none" or prototext_model_fn == "none") {
      if (comm->am_world_master()) {
        cerr << endl << __FILE__ << " " << __LINE__
             << " :: error - you must use  --prototext_fn and "
             << "--prototext_dr_fn to supply prototext filenames\n\n";
      }
      Finalize();
      return 9;
    }

    // read in the prototext files
    lbann_data::LbannPB pb;
    lbann_data::LbannPB pb_reader;
    readPrototextFile(prototext_model_fn.c_str(), pb);
    readPrototextFile(prototext_dr_fn.c_str(), pb_reader);
    lbann_data::Model *pb_model = pb.mutable_model();

    //adjust the prototext with any over-rides
    if (mini_batch_size != 0) {
      pb_model->set_mini_batch_size(mini_batch_size);
    }
    if (num_epochs != 0) {
      pb_model->set_num_epochs(num_epochs);
    }
    //////////////////////////////////////////////////////////////////////
    //NOTE: do not use mini_batch_size or num_epochs, etc, after this block;
    //      instead, use pb_model->mini_batch_size(), etc.
    //////////////////////////////////////////////////////////////////////

    // Set algorithmic blocksize
    SetBlocksize(pb_model->block_size());

    // Set up the communicator and get the grid.
    comm = new lbann_comm(pb_model->procs_per_model());
    Grid& grid = comm->get_model_grid();
    if (comm->am_world_master()) {
      cout << "Number of models: " << comm->get_num_models() << endl;
      cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
      cout << endl;
    }

    // Initialize lbann with the communicator.
    lbann::initialize(comm);
    init_random(42);
    init_data_seq_random(42);

    const lbann_data::DataReader &d_reader = pb.data_reader();
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

    ///////////////////////////////////////////////////////////////////
    // initialize data readers
    //@todo: code not in place for correctly handling image preprocessing
    ///////////////////////////////////////////////////////////////////
    const lbann_data::Model &m2 = pb.model();
    cerr << "calling init_data_readers: " << pb_model->mini_batch_size() << " " << m2.mini_batch_size() << endl << endl;
    std::map<execution_mode, DataReader*> data_readers;
    init_data_readers(comm->am_world_master(), pb_reader, data_readers, pb_model->mini_batch_size());
    if (comm->am_world_master()) {
      for (auto it : data_readers) {
        cerr << "data reader; role: " << it.second->get_role()
             << " num data: " << it.second->getNumData() << endl;
      }
    }
    cerr << "DONE calling init_data_readers: " << pb_model->mini_batch_size() << " " << m2.mini_batch_size() << endl << endl;

    //user feedback
    if (comm->am_world_master()) {
      cout << "\nParameter settings:" << endl;
      cout << "\tMini-batch size: " << pb_model->mini_batch_size() << endl;
      const lbann_data::Optimizer optimizer = pb_model->optimizer();
      cout << "\tLearning rate: " <<  optimizer.learn_rate() << endl;
      cout << "\tEpoch count: " << pb_model->num_epochs() << endl << endl;
    }

    ///////////////////////////////////////////////////////////////////
    // initalize model
    // @todo: not all callbacks code is in place
    ///////////////////////////////////////////////////////////////////
    optimizer_factory *optimizer_fac = init_optimizer_factory(comm, pb);
    cudnn::cudnn_manager* cudnn = NULL;
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
    sequential_model * model = init_model(comm, optimizer_fac, pb);
    add_layers(model, data_readers, cudnn, pb);
    init_callbacks(comm, model, pb);
    model->setup();

    // restart model from checkpoint if we have one
    //@todo
    //model->restartShared();

    ///////////////////////////////////////////////////////////////////
    // main loop for training/testing
    ///////////////////////////////////////////////////////////////////
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
  Finalize();

  return 0;
}
