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
#include <cstdlib>

using namespace lbann;

const int lbann_default_random_seed = 42;

model * build_model_from_prototext(int argc, char **argv, lbann_data::LbannPB &pb);
          

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
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

    std::vector<lbann_data::LbannPB *> pbs;
    protobuf_utils::load_prototext(master, argc, argv, pbs);

    model *model_1 = build_model_from_prototext(argc, argv, *(pbs[0])); //D1 solver
    model *model_2 = nullptr; //G1 solver
    model *model_3 = nullptr; //G2 solver

    if (pbs.size() > 1) {
      model_2 = build_model_from_prototext(argc, argv, *(pbs[1]));
    }

    if (pbs.size() > 2) {
      model_3 = build_model_from_prototext(argc, argv, *(pbs[2]));
    }
    
    const lbann_data::Model pb_model = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();
    const lbann_data::Model pb_model_3 = pbs[2]->model();
    
    int super_step = 1;
    int max_super_step = pb_model.super_steps(); 
    while (super_step <= max_super_step) {
      if (master)  std::cerr << "\nSTARTING train - discriminator (D1 & D2) models at step " << super_step <<"\n\n";
      model_1->train( super_step*pb_model.num_epochs(),pb_model_2.num_batches());

      if(master) std::cout << " Copy all trained weights from model1 to model2 and train/freeze as appropriate " << std::endl;
      auto model1_weights = model_1->get_weights();
      model_2->copy_trained_weights_from(model1_weights);
      if (master) std::cerr << "\n STARTING train - G1 solver model at step " << super_step << " \n\n";
      model_2->train( super_step*pb_model_2.num_epochs(),pb_model_2.num_batches());
      //@todo: remove the dependencies
      if(master) std::cout << " Copy all trained weights from model1 to model3 and train/freeze as appropriate " << std::endl;
      //auto model2_weights = model_2->get_weights();
      model_3->copy_trained_weights_from(model1_weights);
      if (master) std::cerr << "\n STARTING train - G2 solver model at step " << super_step << " \n\n";
      model_3->train( super_step*pb_model_3.num_epochs(),pb_model_3.num_batches());

      if(master) std::cout << " Update generator1 weights " << std::endl;
      auto model2_weights = model_2->get_weights();
      model_1->copy_trained_weights_from(model2_weights);
      if(master) std::cout << " Update generator2 weights " << std::endl;
      auto model3_weights = model_3->get_weights();
      model_1->copy_trained_weights_from(model3_weights);
      super_step++;          
    }

   
   
    delete model_1;
    if (model_2 != nullptr) {
      delete model_2;
    }
    if (model_3 != nullptr) {
      delete model_3;
    }
    for (auto t : pbs) {
      delete t;
    }

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (std::exception& e) {
    El::ReportException(e);  // Elemental exceptions
  }

  // free all resources by El and MPI
  finalize(comm);
  return 0;
}
   
model * build_model_from_prototext(int argc, char **argv, lbann_data::LbannPB &pb) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  if (master) std::cerr << "starting build_model_from_prototext\n";
  model *model = nullptr; //d hysom bad namimg! should fix
  try {
    std::stringstream err;
    options *opts = options::get();

    // Optionally over-ride some values in prototext
    get_cmdline_overrides(comm, pb);

    lbann_data::Model *pb_model = pb.mutable_model();

    // Adjust the number of parallel readers; this may be adjusted
    // after calling split_models()
    set_num_parallel_readers(comm, pb);

    // Set algorithmic blocksize
    if (pb_model->block_size() == 0 and master) {
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
    //save_session(comm, argc, argv, pb);

    // Check for cudnn, with user feedback
    cudnn::cudnn_manager *cudnn = nullptr;
#ifdef LBANN_HAS_CUDNN
    if (! pb_model->disable_cuda()) {
      if (master) {
        std::cerr << "code was compiled with LBANN_HAS_CUDNN, and we are using cudnn\n";
      }
      if(pb_model->use_nccl()) {
        cudnn = new cudnn::cudnn_manager(comm, pb_model->num_gpus(), true);
      }
      else{
        cudnn = new cudnn::cudnn_manager(comm, pb_model->num_gpus(), false);
      }
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
        std::cout << "  GPUs on node                 : " << cudnn->get_num_visible_gpus() << std::endl
                  << "  GPUs per process             : " << cudnn->get_num_gpus() << std::endl;

        const char* mv2_ptr = std::getenv("MV2_USE_CUDA");
        const std::string mv2_str = (mv2_ptr == nullptr)? "" : std::string(mv2_ptr);
        std::cout << "  MV2_USE_CUDA                 : " << mv2_str << std::endl;
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
    model = proto::construct_model(comm,
                                   cudnn,
                                   data_readers,
                                   pb.optimizer(),
                                   pb.model());
    model->setup();

    // restart model from checkpoint if we have one
    //@todo
    //model->restartShared();

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

#ifndef LBANN_SEQUENTIAL_CONSISTENCY
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

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  } catch (std::exception& e) {
    El::ReportException(e);  // Elemental exceptions
  }

  return model;
}
