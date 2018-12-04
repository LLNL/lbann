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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann_library.hpp"
#include <dirent.h>
#include <cstdlib>

namespace lbann {

model * __attribute__((used)) build_model_from_prototext(int argc, char **argv,
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

    // Set up the communicator and get the grid based on the first model's spec.
    // We do not currently support splitting different models in different ways,
    // as this implies different grids.
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
      print_lbann_configuration(pb_model, comm);
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

    //under development; experimental
    if (opts->has_bool("use_data_store") && opts->get_bool("use_data_store")) {
      if (master) {
        std::cerr << "\nUSING DATA STORE!\n\n";
      }
      for (auto r : data_readers) {
        if (!r.second) continue;
        r.second->setup_data_store(model);
      }
    }

    if (opts->has_string("create_tarball")) {
      finalize(comm);
      return 0;
    }

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

  } catch (lbann_exception& e) {
    El::mpi::Abort(El::mpi::COMM_WORLD, 1);
  } catch (std::exception& e) {
    El::ReportException(e);  // Elemental exceptions
  }

  return model;
}

bool __attribute__((used)) load_model_weights(std::string ckpt_dir, model * m){
  std::vector<std::string> weight_list = std::vector<std::string>();
  int epochLast = -1;
  int stepLast = -1;
  // define filename
  char latest[1024];
  sprintf(latest, "%s/last.shared.checkpoint", ckpt_dir.c_str());
  // get last epoch and step saved.
  int fd = openread(latest);
  lbann_comm *temp_comm = m->get_comm();
  if (fd != -1) {
    char field[256];
    read_string(fd, "shared.last", field, sizeof(field));
    int ret = sscanf(field, "epoch=%d step=%d\n", &epochLast, &stepLast);
    if(ret != 2) { return false; }
    closeread(fd, latest);
    if(temp_comm->am_model_master())
      sprintf(latest, "%s/shared.model.%d.epoch.%d.step.%d/", ckpt_dir.c_str(), temp_comm->get_model_rank(), epochLast, stepLast);
    temp_comm->model_broadcast(0, &(latest[0]), sizeof(latest), El::SyncInfo<El::Device::CPU>{});
  }

  DIR *weight_dir;
  struct dirent *weight_file;
  if((weight_dir = opendir(latest)) == NULL)
  {
    std::cout << "error opening " << latest << "\n";
    return false;
  }
  // Populate weight list
  while ((weight_file = readdir(weight_dir)) != NULL){
    if(!strncmp(weight_file->d_name,"model_weights_",14))
      weight_list.push_back(std::string(weight_file->d_name));
  }
  closedir(weight_dir);
  // load weights that appear in weight list.
  for(weights *w : m->get_weights()) {
    w->load_from_save(latest,weight_list);
  }
  return true;
}

void print_lbann_configuration(lbann_data::Model *pb_model, lbann_comm *comm) {
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
  const auto& grid = comm->get_model_grid();
  int procs_per_model = pb_model->procs_per_model();
  std::cout << "Model settings" << std::endl
            << "  Models              : " << comm->get_num_models() << std::endl
            << "  Processes per model : " << procs_per_model << std::endl
            << "  Grid dimensions     : " << grid.Height() << " x " << grid.Width() << std::endl;
  std::cout << std::endl;
}

} // namespace lbann
