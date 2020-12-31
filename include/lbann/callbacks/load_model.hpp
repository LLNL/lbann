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

#ifndef LBANN_CALLBACKS_CALLBACK_LOAD_MODEL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_LOAD_MODEL_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

#include <google/protobuf/message.h>

// Forward-declare protobuf classes
namespace lbann_data {
class Model;
}

namespace lbann {
namespace callback {

/**
 * Load pretrained model from file
 */
class load_model : public callback_base {
 public:
  /**
   * @param dir directory to load model
   * @param extension file extension e.g., model, state ......
   */
  load_model(std::vector<std::string> dirs,
             std::string extension="prototext") :
    callback_base(), m_dirs(std::move(dirs)),
    m_extension(std::move(extension)),
    m_loaded(false)
  {}
  load_model(const load_model&) = default;
  load_model& operator=(
    const load_model&) = default;
  load_model* copy() const override {
    return new load_model(*this);
  }

  inline void add_dir(const std::string& dir){
    m_dirs.emplace_back(dir);
  }

  void on_train_begin(model *m) override;

  void on_test_begin(model *m) override;

  std::string name() const override { return "load model"; }

  /** @name Checkpointing */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    ar(cereal::base_class<callback_base>(this),
       CEREAL_NVP(m_dirs),
       CEREAL_NVP(m_extension),
       CEREAL_NVP(m_loaded));
  }

  ///@}

 protected:
  friend class lbann::model;


 private:
  std::vector<std::string> m_dirs; //director(ies) to load pretrained model(s)
  /// Disables the normal behavior of saving when training is complete
  std::string m_extension; //file extension

  /// Flag to indicate if the model has already been loaded
  bool m_loaded;
};

// Builder function
std::unique_ptr<callback_base>
build_load_model_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_LOAD_MODEL_HPP_INCLUDED
