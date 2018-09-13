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

model * build_model_from_prototext(int argc, char **argv,
                                   lbann_data::LbannPB &pb,
                                   lbann_comm *comm,
                                   bool first_model);


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

    model *model_1 = build_model_from_prototext(argc, argv, *(pbs[0]),
                                                comm, true); //D1 solver
    model *model_2 = nullptr; //G1 solver
    model *model_3 = nullptr; //G2 solver

    //Support for autoencoder models
    model *ae_model = nullptr;  
    model *ae_cycgan_model = nullptr; //contain layer(s) from (cyc)GAN

    if (pbs.size() > 1) {
      model_2 = build_model_from_prototext(argc, argv, *(pbs[1]),
                                           comm, false);
    }

    if (pbs.size() > 2) {
      model_3 = build_model_from_prototext(argc, argv, *(pbs[2]),
                                           comm, false);
    }
     
    if (pbs.size() > 3) {
      ae_model = build_model_from_prototext(argc, argv, *(pbs[3]),
                                           comm, false);
    }

    if (pbs.size() > 4) {
      ae_cycgan_model = build_model_from_prototext(argc, argv, *(pbs[4]),
                                           comm, false);
    }

    const lbann_data::Model pb_model = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();
    const lbann_data::Model pb_model_3 = pbs[2]->model();

    //Optionally pretrain autoencoder
    //@todo: explore joint-train of autoencoder as alternative
    if(ae_model != nullptr) {
      if(master) std::cout << " Pre-train autoencoder " << std::endl;
      const lbann_data::Model pb_model_4 = pbs[3]->model();
      ae_model->train(pb_model_4.num_epochs());
      auto ae_weights = ae_model->get_weights();
      model_1->copy_trained_weights_from(ae_weights);
      model_2->copy_trained_weights_from(ae_weights);
      model_3->copy_trained_weights_from(ae_weights);
      ae_cycgan_model->copy_trained_weights_from(ae_weights);
    }
    
    //Train cycle GAN
    int super_step = 1;
    int max_super_step = pb_model.super_steps();
    while (super_step <= max_super_step) {
      if (master)  std::cerr << "\nSTARTING train - discriminator (D1 & D2) models at step " << super_step <<"\n\n";
      model_1->train( super_step*pb_model.num_epochs(),pb_model_2.num_batches());

      if(master) std::cout << " Copy all trained weights from discriminator to G1 and train/freeze as appropriate " << std::endl;
      auto model1_weights = model_1->get_weights();
      model_2->copy_trained_weights_from(model1_weights);
      if (master) std::cerr << "\n STARTING train - G1 solver model at step " << super_step << " \n\n";
      model_2->train( super_step*pb_model_2.num_epochs(),pb_model_2.num_batches());
      // Evaluate model on test set
      model_2->evaluate(execution_mode::testing);

      if(master) std::cout << " Copy all trained weights from discriminator to G2 and train/freeze as appropriate " << std::endl;
      model_3->copy_trained_weights_from(model1_weights);
      if (master) std::cerr << "\n STARTING train - G2 solver model at step " << super_step << " \n\n";
      model_3->train( super_step*pb_model_3.num_epochs(),pb_model_3.num_batches());
      // Evaluate model on test set
      model_3->evaluate(execution_mode::testing);

      if(master) std::cout << " Update G1 weights " << std::endl;
      auto model2_weights = model_2->get_weights();
      model_1->copy_trained_weights_from(model2_weights);
      if(master) std::cout << " Update G2 weights " << std::endl;
      auto model3_weights = model_3->get_weights();
      model_1->copy_trained_weights_from(model3_weights);
      
      //Optionally evaluate on pretrained autoencoder
      if(ae_model != nullptr && ae_cycgan_model != nullptr){
        //if(master) std::cout << " Copy trained weights from autoencoder to autoencoder proxy" << std::endl;
        //ae_cycgan_model->copy_trained_weights_from(ae_weights);
        if(master) std::cout << " Copy trained weights from cycle GAN" << std::endl;
        ae_cycgan_model->copy_trained_weights_from(model2_weights);
        if(master) std::cout << " Evaluate pretrained autoencoder" << std::endl;
        ae_cycgan_model->evaluate(execution_mode::testing);
       }

      super_step++;
    }



    delete model_1;
    if (model_2 != nullptr) {
      delete model_2;
    }
    if (model_3 != nullptr) {
      delete model_3;
    }
    if (ae_model != nullptr) {
      delete ae_model;
    }
    if (ae_cycgan_model != nullptr) {
      delete ae_cycgan_model;
    }
    for (auto t : pbs) {
      delete t;
    }

  } catch (std::exception& e) {
    El::ReportException(e);
  }

  // free all resources by El and MPI
  finalize(comm);
  return 0;
}

model * build_model_from_prototext(int argc, char **argv,
                                   lbann_data::LbannPB &pb,
                                   lbann_comm *comm,
                                   bool first_model) {
  int random_seed = lbann_default_random_seed;
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
    // Initialize models differently if needed.
#ifndef LBANN_DETERMINISTIC
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
    if (first_model) {
      comm->split_models(procs_per_model);
      if (pb_model->num_parallel_readers() > procs_per_model) {
        pb_model->set_num_parallel_readers(procs_per_model);
      }
    } else if (procs_per_model != comm->get_procs_per_model()) {
      LBANN_ERROR("Model prototexts requesting different procs per model is not supported");
    }

    // Save info to file; this includes the complete prototext (with any over-rides
    // from the cmd line) and various other info
    save_session(comm, argc, argv, pb);

    // Report useful information
    if (master) {

      // Report hardware settings
      std::cout << "Hardware properties (for master process)" << std::endl
                << "  Processes on node          : " << comm->get_procs_per_node() << std::endl
                << "  OpenMP threads per process : " << omp_get_max_threads() << std::endl;
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
      std::cout << "  Aluminum : ";
#ifdef LBANN_HAS_ALUMINUM
      std::cout << "detected" << std::endl;
#else
      std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_ALUMINUM
#endif // LBANN_DEBUG
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
      const auto& grid = comm->get_model_grid();
      std::cout << "Model settings" << std::endl
                << "  Models              : " << comm->get_num_models() << std::endl
                << "  Processes per model : " << procs_per_model << std::endl
                << "  Grid dimensions     : " << grid.Height() << " x " << grid.Width() << std::endl;
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

    // hack to prevent all data readers from loading identical data; instead,
    // share a single copy. See data_reader_jag_conduit_hdf5 for example
    if (first_model) {
      if (opts->has_string("share_data_reader_data")) {
        for (auto t : data_readers) {
          opts->set_ptr((void*)t.second);
        }
      }
    }

    // User feedback
    print_parameters(comm, pb);

    // Initalize model
    model = proto::construct_model(comm,
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

  } catch (std::exception& e) {
    El::ReportException(e);
  }

  return model;
}
