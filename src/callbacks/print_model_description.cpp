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

#include "lbann/callbacks/print_model_description.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/serialize.hpp"
#include <callbacks.pb.h>

namespace lbann {
namespace callback {

template <class Archive>
void print_model_description::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)));
}

void print_model_description::on_setup_end(model *m) {
  if (m->get_comm()->am_world_master()) {
    std::cout << "\n"
              << m->get_description()
              << std::endl;
  }
}

std::unique_ptr<callback_base>
build_print_model_description_callback_from_pbuf(
  const google::protobuf::Message&, const std::shared_ptr<lbann_summary>&) {
  return make_unique<print_model_description>();
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::print_model_description
#define LBANN_CLASS_LIBNAME callback_print_model_description
#include <lbann/macros/register_class_with_cereal.hpp>
