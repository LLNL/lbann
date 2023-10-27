////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "lbann/proto/callbacks.pb.h"
#include "lbann/proto/model.pb.h"

#include <cstdlib>
#include <fstream>
#include <string>

namespace lbann {
namespace {

auto make_weights_map(std::vector<weights*> const& weights_list)
{
  std::unordered_map<std::string, weights*> weights_map;
  for (auto* w : weights_list)
    weights_map.emplace(w->get_name(), w);
  return weights_map;
}

void load_weights_from_checkpoint(model& m, std::string const& model_ckpt_file)
{
  auto comm = m.get_comm();
  // TODO: Logging API
  if (comm->am_trainer_master()) {
    std::cout << "Restoring from " << model_ckpt_file << std::endl;
  }

  // Create a temporary model
  model tmp_model(comm, nullptr, nullptr);
  {
    std::ifstream ifs(model_ckpt_file);
    RootedBinaryInputArchive ar(ifs, comm->get_trainer_grid());
    ar(tmp_model);
  }

  // Loop through the weights in this model and attempt to restore
  // their values from the temporary model's weights with the same
  // name.
  auto tmp_weights_map = make_weights_map(tmp_model.get_weights());
  auto const model_weights = m.get_weights();
  for (auto* w : model_weights) {
    auto tmp_w_iter = tmp_weights_map.find(w->get_name());
    if (tmp_w_iter != tmp_weights_map.end()) {
      auto* tmp_w = tmp_w_iter->second;
      w->steal_values(*tmp_w);
      // TODO: Replace with logging API
      if (comm->am_trainer_master()) {
        std::cout << "Restored weights \"" << w->get_name() << "\" "
                  << "from checkpointed model." << std::endl;
      }
    }
    else {
      // TODO: Replace with logging API
      if (comm->am_trainer_master()) {
        std::cout << "Could not load weights with name \"" << w->get_name()
                  << "\". Not found in checkpoint." << std::endl;
      }
    }
  }
}

void load_weights_from_files(model& m, std::string const& ckpt_dir)
{
  auto const model_weights = m.get_weights();
  for (weights* w : model_weights) {
    // create weight file name to match to weight list entry
    auto* dtw = dynamic_cast<data_type_weights<DataType>*>(w);
    LBANN_ASSERT(dtw);

    auto const file =
      file::join_path(ckpt_dir,
                      build_string("model_weights_",
                                   w->get_name(),
                                   "_",
                                   dtw->get_values_sharded().Height(),
                                   "x",
                                   dtw->get_values_sharded().Width(),
                                   ".bin"));
    if (file::file_exists(file)) {
      // TODO: Replace with logging API
      if (m.get_comm()->am_trainer_master()) {
        std::cout << "Loading: " << file << std::endl;
      }
      El::Read(dtw->get_values_sharded(), file, El::BINARY, true);
    }
    else {
      // TODO: Replace with logging API
      if (m.get_comm()->am_trainer_master()) {
        std::cout << "Could not load weights with name \"" << w->get_name()
                  << "\". Expected file not found (" << file << ")."
                  << std::endl;
      }
    }
  }
}

// (trb 12/30/2020): My understanding is that `m` should be a
// constructed model with a DAG and weights that at least have
// names. This function will then loop through the weights objects of
// `m` and look for a corresponding weights object in the appropriate
// directory.
//
// Weights can be restored from independent files storing the binary
// weights or from a checkpoint. In the first case, weights are
// matched by filename. In the latter case, the entire model is
// restored into a temporary model object, which is then stripped for
// parts and discarded. If dedicated weights files are sharing a
// directory with a checkpoint, the weights will be pulled from the
// checkpoint. (In the future, it might be faster to prefer the
// standalone weights objects, but testing for `model.bin` is
// faster/easier than checking each `<weights_name>.bin`.)
bool load_model_weights(const std::string& ckpt_dir, model& m)
{
  std::string const active_ckpt_dir = add_delimiter(ckpt_dir);
  LBANN_ASSERT(file::directory_exists(active_ckpt_dir));

  // TODO: Replace with logging API
  if (m.get_comm()->am_trainer_master()) {
    std::cout << "Loading model weights from " << active_ckpt_dir << std::endl;
  }

  auto const checkpoint_file = file::join_path(active_ckpt_dir, "model.bin");
  if (file::file_exists(checkpoint_file))
    load_weights_from_checkpoint(m, checkpoint_file);
  else
    load_weights_from_files(m, active_ckpt_dir);
  return true;
}
} // namespace

namespace callback {

template <class Archive>
void load_model::serialize(Archive& ar)
{
  ar(cereal::base_class<callback_base>(this),
     CEREAL_NVP(m_dirs),
     CEREAL_NVP(m_extension),
     CEREAL_NVP(m_loaded));
}

void load_model::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_load_model();
  msg->set_dirs(protobuf::to_space_sep_string(m_dirs));
  msg->set_extension(m_extension);
}
void load_model::on_train_begin(model* m)
{
  if (!m_loaded) {
    for (const auto& d : m_dirs) {
      m_loaded = load_model_weights(d, *m);
      if (!m_loaded)
        LBANN_ERROR("Unable to reload model on train begin");
    }
  }
}

void load_model::on_test_begin(model* m)
{
  if (!m_loaded) {
    for (const auto& d : m_dirs) {
      m_loaded = load_model_weights(d, *m);
      if (!m_loaded)
        LBANN_ERROR("Unable to reload model on test begin");
    }
  }
}

std::unique_ptr<callback_base>
build_load_model_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                                    const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackLoadModel&>(proto_msg);
  if (params.extension().size() != 0) {
    return std::make_unique<load_model>(parse_list<std::string>(params.dirs()),
                                        params.extension());
  }
  else {
    return std::make_unique<load_model>(parse_list<std::string>(params.dirs()));
  }
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::load_model
#define LBANN_CLASS_LIBNAME callback_load_model
#include <lbann/macros/register_class_with_cereal.hpp>
