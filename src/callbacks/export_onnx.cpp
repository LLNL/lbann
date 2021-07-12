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
{}

void export_onnx::on_setup_end(model* m)
{
  mp_.set_ir_version(7);
  //FIXME: what goes here? ONNX operators?
  // The empty string ("") domain indicates the operators defined
  // as part of the ONNX specification; other domains correspond
  // to operator sets of other vendors (e.g., they can be used to
  // provide vendor-specific extensions to ONNX)
  auto* opset = mp_.add_opset_import();
  opset->set_domain("");
  opset->set_version(11);

  mp_.set_producer_name("LBANN");
  mp_.set_producer_version(LBANN_MAKE_STR(LBANN_VERSION));
  mp_.set_domain("lbann/LLNL/com.github");
  // FIXME: model version is version of graph encoded.
  //        Should this be passed in?
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
  // FIXME: Use gp->initializer for for weights ?

  auto const layers = m->get_layers();
  for (auto const* layer : layers)
  {
    layer->fill_onnx_node(*gp);
  }

  // FIXME: We don't use sparse_initializer. Do we need to handle it since
  //        its a message type or just ignore it?
  /*
  auto* sparse_initializer = gp->add_sparse_initializer();
  auto* sparse_values = sparse_initializer->mutable_values();
  for( auto dim : dims )
    sparse_values->add_dims(dim);
  sparse_values->set_data_type(0);
  auto* sparse_indices = sparse_initializer->mutable_indices();
  for( auto dim : dims )
    sparse_indices->add_dims(dim);
  sparse_indices->set_data_type(0);
  */

  // FIXME: Name, layers, get_type
  gp->set_doc_string(m->get_name());

  // ValueInfoProto input will be filled with input layer info in
  //    overridden fill_onnx_node func. in input_layer.hpp

  // FIXME: Used in eval layer
  // auto* output = gp->add_output();

  // FIXME: Not useful for now
  // auto* value_info = gp->add_value_info();

  // Not useful for now
  // auto* quantization_annotation = gp->add_quantization_annotation();

  std::cout << mp_.DebugString() << std::endl;

  std::ofstream onnx_out("./test_output.onnx");
  mp_.SerializeToOstream(&onnx_out);

}

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer) {

  return make_unique<export_onnx>(summarizer);
}
}// namespace callback
}// namespace lbann
