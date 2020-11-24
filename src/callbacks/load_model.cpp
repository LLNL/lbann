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
//
// load_model .hpp .cpp - Callbacks to load pretrained model(s)
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/load_model.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/training_algorithms/training_algorithm.hpp"

#include <callbacks.pb.h>
#include <model.pb.h>


#include <unistd.h>
#include <dirent.h>

#include <cstdlib>
#include <fstream>
#include <string>

namespace lbann {
namespace callback {


void load_model::on_train_begin(model *m) {
  if(!m_loaded) {
    for (const auto& d : m_dirs) {
      m_loaded = load_model_weights(d, "", m, true);
      if(!m_loaded)  LBANN_ERROR("Unable to reload model on train begin");
    }
  }
}

void load_model::on_test_begin(model *m) {
  if(!m_loaded) {
    for (const auto& d : m_dirs) {
      m_loaded = load_model_weights(d, "", m, true);
      if(!m_loaded)  LBANN_ERROR("Unable to reload model on test begin");
    }
  }
}


bool load_model::load_model_weights(const std::string& ckpt_dir,
                                    const std::string& alg_name,
                                    model *m,
                                    bool ckptdir_is_fullpath) {
  std::vector<std::string> weight_list = std::vector<std::string>();
  std::string active_ckpt_dir;
  if(ckptdir_is_fullpath) {
    active_ckpt_dir = add_delimiter(ckpt_dir);
  }else {
    size_t epochLast = std::numeric_limits<size_t>::max();;
    size_t stepLast = std::numeric_limits<size_t>::max();;
    execution_mode mode = execution_mode::invalid;
    active_ckpt_dir = get_last_shared_checkpoint_filename(alg_name, ckpt_dir);

    // get last epoch and step saved.
    int success = read_latest(active_ckpt_dir, &mode, &epochLast, &stepLast);
    if(!success) {
      LBANN_WARNING("Unable to find the latest checkpoint ", active_ckpt_dir);
      return false;
    }
    active_ckpt_dir = get_shared_checkpoint_dirname(alg_name, ckpt_dir, mode, epochLast, stepLast) + m->get_name() + '/';
  }

  lbann_comm *comm = m->get_comm();
  if(comm->am_trainer_master()) {
    std::cout << "Loading model weights from " << active_ckpt_dir << std::endl;
  }

  DIR *weight_dir = opendir(active_ckpt_dir.c_str());
  if(weight_dir == nullptr)
  {
    LBANN_WARNING("error opening ",  active_ckpt_dir);
    return false;
  }
  // Populate weight list
  struct dirent *weight_file;
  while ((weight_file = readdir(weight_dir)) != nullptr){
    if(!strncmp(weight_file->d_name,"model_weights_",14))
      weight_list.push_back(std::string(weight_file->d_name));
  }
  closedir(weight_dir);

  // load weights that appear in weight list.
  // load weights that appear in weight list.
  // for(auto&& w : m_weights) {
  //   w->load_from_save(latest,weight_list);
  // }
  //  m->reload_weights(active_ckpt_dir, weight_list);
  return true;
}

std::unique_ptr<callback_base>
build_load_model_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackLoadModel&>(proto_msg);
  if(params.extension().size() != 0) {
    return make_unique<load_model>(
      parse_list<std::string>(params.dirs()),
      params.extension());
  }
  else {
    return make_unique<load_model>(
      parse_list<std::string>(params.dirs()));
  }
}

} // namespace callback
} // namespace lbann
