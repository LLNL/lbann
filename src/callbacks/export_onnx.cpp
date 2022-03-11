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

#include <callbacks.pb.h>

#include <fstream>
#include <iostream>
#include <string>

namespace lbann {
namespace callback {

void export_onnx::on_train_end(model* m)
{
  auto const rank = m->get_comm()->get_rank_in_trainer();

  m->serialize_to_onnx(mp_);

  std::cout << "OUTPUT=" << m_output_filename << ", DEBUG="
            << m_debug_string_filename << std::endl;

  if (rank == 0) {
    std::ofstream onnx_out(m_output_filename);
    mp_.SerializeToOstream(&onnx_out);

    if (m_debug_string_filename != "") {
      std::ofstream debug(m_debug_string_filename);
      debug << mp_.DebugString();
    }
  }
}

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                                     const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackExportOnnx&>(proto_msg);
  return std::make_unique<export_onnx>(
    (params.output_filename().size() == 0
     ? std::string("lbann_output.onnx") : params.output_filename(),
     params.debug_string_filename().size() == 0
     ? std::string("") : params.debug_string_filename()));
}
} // namespace callback
} // namespace lbann
