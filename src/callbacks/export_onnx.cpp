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
// export_onnx .hpp .cpp - Exports trained model to onnx format
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/export_onnx.hpp"

#include "lbann/layers/io/input_layer.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/summary_impl.hpp"

#include <callbacks.pb.h>

#include <fstream>
#include <iostream>
#include <string>


namespace lbann {
namespace callback {

export_onnx::export_onnx(bool print_debug_string,
                         std::string output_file)
  : callback_base(/*batch_interval=*/1),
    m_print_debug_string(print_debug_string),
    m_output_file(output_file)
{}

void export_onnx::on_setup_end(model* m)
{
  mp_.set_ir_version(7);
  auto* opset = mp_.add_opset_import();
  // The empty string ("") domain indicates the operators defined
  // as part of the ONNX specification; other domains correspond
  // to operator sets of other vendors (e.g., they can be used to
  // provide vendor-specific extensions to ONNX)
  opset->set_domain("");
  opset->set_version(11);

  mp_.set_producer_name("LBANN");
  mp_.set_producer_version(LBANN_MAKE_STR(LBANN_VERSION));
  mp_.set_domain("lbann/LLNL/com.github");
  mp_.set_model_version(1);
  mp_.set_doc_string("Livermore Big Artificial Neural Network");
}

void export_onnx::on_train_begin(model* m)
{
  // graph info
  auto* gp = mp_.mutable_graph();
  gp->set_name(m->get_name());

  auto const layers = m->get_layers();
  for (auto const* layer : layers) {
    layer->fill_onnx_node(*gp);
  }
  gp->set_doc_string(m->get_name());

  auto rank = m->get_comm()->get_rank_in_trainer();
  if( rank == 0 ) {
    std::ofstream onnx_out(m_output_file);
    mp_.SerializeToOstream(&onnx_out);

    if(m_print_debug_string)
      std::cout << mp_.DebugString() << std::endl;
  }
}

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackExportOnnx&>(proto_msg);
  return std::make_unique<export_onnx>(
    params.print_debug_string(),
    params.output_file());
}
}// namespace callback
}// namespace lbann
