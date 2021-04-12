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
//#include <catch2/catch.hpp>

#include <iostream>
#include "lbann/callbacks/export_onnx.hpp"

#include "lbann/layers/io/input_layer.hpp"

#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/summary_impl.hpp"

#include <callbacks.pb.h>


namespace lbann {
namespace callback {

export_onnx::export_onnx(std::shared_ptr<lbann_summary> const& summarizer)
  : m_summarizer(summarizer)
{
  std::cout << "I do stuff!" << std::endl;
}

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer) {

  return make_unique<export_onnx>(summarizer);
}
}// namespace callback
}// namespace lbann
