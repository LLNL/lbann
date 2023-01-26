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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_PRINT_MODEL_DESCRIPTION_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PRINT_MODEL_DESCRIPTION_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/** @brief Print human-readable description of model to standard input.
 *
 *  Message is printed when the model has finished setup. The
 *  description includes information on the model's layers, weights,
 *  and callbacks.
 */
class print_model_description : public callback_base {
public:
  print_model_description() : callback_base() {}
  print_model_description(const print_model_description&) = default;
  print_model_description& operator=(const print_model_description&) = default;
  print_model_description* copy() const override { return new print_model_description(*this); }
  void on_setup_end(model *m) override;
  std::string name() const override { return "print_model_description"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

};

// Builder function
std::unique_ptr<callback_base>
build_print_model_description_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_PRINT_MODEL_DESCRIPTION_HPP_INCLUDED
