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
// export_onnx .hpp .cpp - Exports trained model to onnx format
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>

#include "lbann/callbacks/export_onnx.hpp"
#include "lbann/layers/io/input_layer.hpp"

#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/summary_impl.hpp"

#include <callbacks.pb.h>

#include <string>


namespace lbann {
namespace callback {

export_onnx::export_onnx(std::shared_ptr<lbann_summary> const& summarizer)
  : m_summarizer(summarizer)
{
#ifndef LBANN_HAS_ONNX
  std::cout << "TESTING CALLBACK: ONNX NOT DEF " << std::endl;
#endif
}

#ifdef LBANN_HAS_ONNX
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
  // FIXME: what should go here??
  auto* metadata = mp_.add_metadata_props();
  metadata->set_key("name of thing");
  metadata->set_value("thing");
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

  // FIXME: Name, layers, get_type
  std::string model_name = "Model Name: " + m->get_name() + ", ";
  std::string layer_names = "Model Layers: ";
  for( auto const* layer : layers) {
    layer_names.append(layer->get_name() + ", ");
  }
  std::string model_type = "Model Type: " + m->get_type();
  gp->set_doc_string(model_name + layer_names + model_type);

  std::cout << mp_.DebugString() << std::endl;

  std::ofstream onnx_out("./test_output.onnx");
  mp_.SerializeToOstream(&onnx_out);

}
#endif // LBANN_HAS_ONNX

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer) {

  return make_unique<export_onnx>(summarizer);
}
}// namespace callback
}// namespace lbann
