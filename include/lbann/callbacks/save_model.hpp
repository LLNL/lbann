////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
// save_model .hpp .cpp - Callbacks to save model, currently as protobuf
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SAVE_MODEL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SAVE_MODEL_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/io/persist.hpp"

#include <google/protobuf/message.h>

// Forward-declare protobuf classes
namespace lbann_data {
class Model;
}

namespace lbann {
namespace callback {

/**
 * Save model to as protobuf file and set of weights
 */
class save_model : public callback_base {
 public:
  /**
   * @param dir directory to save model
   * @param disable_save_after_training Don't save after training
   * @param extension file extension e.g., model, state ......
   */
  save_model(std::string dir,
                            bool disable_save_after_training,
                            std::string extension="prototext") :
    callback_base(), m_dir(std::move(dir)),
    m_disable_save_after_training(disable_save_after_training),
    m_extension(std::move(extension))
  {}
  save_model(const save_model&) = default;
  save_model& operator=(
    const save_model&) = default;
  save_model* copy() const override {
    return new save_model(*this);
  }
  void on_train_end(model *m) override;
  std::string name() const override { return "save model"; }
  void set_target_dir(const std::string& dir) { m_dir = dir; }
  const std::string& get_target_dir() { return m_dir; }

 protected:
  friend class lbann::model;

  bool do_save_model(model *m);
  bool do_save_model_weights(model *m);

 private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  std::string m_dir; //directory to save file
  /// Disables the normal behavior of saving when training is complete
  bool m_disable_save_after_training;
  std::string m_extension; //file extension
  persist p;

  void write_proto_binary(const lbann_data::Model& proto, const std::string filename);
  void write_proto_text(const lbann_data::Model& proto, const std::string filename);
};

inline std::string get_save_model_dirname(const std::string& trainer_name, const std::string& model_name, const std::string& dir) {
  return build_string(dir, '/', trainer_name, '/', model_name, '/');
}

// Builder function
std::unique_ptr<callback_base>
build_save_model_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SAVE_MODEL_HPP_INCLUDED
