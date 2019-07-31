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
// lbann_callback_save_model .hpp .cpp - Callbacks to save model, currently as protobuf
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SAVE_MODEL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SAVE_MODEL_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"
#include <lbann.pb.h>
#include <google/protobuf/message.h>

namespace lbann {

/**
 * Save model to as protobuf file and set of weights
 */
class lbann_callback_save_model : public lbann_callback {
 public:
  /**
   * @param dir directory to save model
   * @param disable_save_after_training Don't save after training
   * @param extension file extension e.g., model, state ......
   */
  lbann_callback_save_model(std::string dir,
                            bool disable_save_after_training,
                            std::string extension="prototext") :
    lbann_callback(), m_dir(std::move(dir)),
    m_disable_save_after_training(disable_save_after_training),
    m_extension(std::move(extension))
    {}
  lbann_callback_save_model(const lbann_callback_save_model&) = default;
  lbann_callback_save_model& operator=(
    const lbann_callback_save_model&) = default;
  lbann_callback_save_model* copy() const override {
    return new lbann_callback_save_model(*this);
  }
  void on_train_end(model *m) override;
  bool save_model(model *m);
  bool save_model_weights(model *m);
  /* ckptdir_is_fullpath flag if true
 * allow user to specify full path to model weights to load
 * and allow system to ignore appending trainer id, num of epochs/steps
 * to default ckpt_dir*/
  static bool load_model_weights(std::string ckpt_dir, model *m, bool ckptdir_is_fullpath=false);

  std::string name() const override { return "save model"; }
 private:
  std::string m_dir; //directory to save file
  bool m_disable_save_after_training; /// Disables the normal behavior of saving when training is complete
  std::string m_extension; //file extension
  persist p;
  void write_proto_binary(const lbann_data::Model& proto, const std::string filename);
  void write_proto_text(const lbann_data::Model& proto, const std::string filename);
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_save_model_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SAVE_MODEL_HPP_INCLUDED
