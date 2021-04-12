////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
// export_onnx .hpp .cpp - Exports trained model to onnx format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_EXPORT_ONNX_HPP_INCLUDED
#define LBANN_CALLBACKS_EXPORT_ONNX_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <lbann/base.hpp>

#include <onnx/onnx_pb.h>

#include <google/protobuf/message.h>

#include <iostream>
#include <memory>
#include <vector>

namespace lbann {
namespace callback {

/** @class export_onnx
 *  @brief Callback to export a trained model to onnx format
 */
class export_onnx : public callback_base {

public:
  /** @brief export_onnx Constructor.
   *  @param output_file Output filename (default = lbann_output.onnx)
   *  @param print_debug_string Option to print debug string to stdout
   */
  export_onnx(bool print_debug_string = false,
              std::string output_file = "lbann_output.onnx");

  /** @brief Copy interface */
  export_onnx* copy() const override {
    return new export_onnx(*this);
  }

  /** @brief Return name of callback */
  std::string name() const override { return "export_onnx"; }

  /* @brief gather model info */
  void on_setup_end(model* m) override;

  /* @brief gather graph/layer info */
  void on_train_begin(model* m) override;

private:

  /* @brief option to print onnx debug string */
  bool m_print_debug_string;

  /* @brief name of output file. Default = lbann_output.onnx */
  std::string m_output_file;

  /* @brief onnx ModelProto object */
  onnx::ModelProto mp_;

}; // class export_onnx

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_EXPORT_ONNX_HPP_INCLUDED
