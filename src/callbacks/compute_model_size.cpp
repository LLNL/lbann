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

#include "lbann/comm_impl.hpp"
#include "lbann/callbacks/compute_model_size.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include <callbacks.pb.h>


namespace lbann {
namespace callback {

compute_model_size::compute_model_size(std::string output_name,
                           El::Int batch_interval)
  : callback_base(batch_interval),
    m_output_name(std::move(output_name)) {
    m_output = nullptr;
}

compute_model_size::compute_model_size()
  : compute_model_size("",0)
{}

template <class Archive>
void compute_model_size::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_output_name));
}

void compute_model_size::setup(model* m) {
   for (auto* w : m->get_weights()) {
      if(w->get_name() == m_output_name){
        m_output = w;
        break;
      }
   }
    if (m_output == nullptr) {
      LBANN_ERROR("Current implementation of callback \"", name(), "\" "
                  "requires a weight object to store computed value");
    }
}

void compute_model_size::on_batch_begin(model* m) {
  const auto& c = m->get_execution_context();
  if (m_output != nullptr && 
      c.get_step() % m_batch_interval == 0 &&
      c.get_step() > 0) {
    compute_size(*m);
  }
}

void compute_model_size::compute_size(model& m){
  size_t model_size = 0; //size_t
  for (auto* w : m.get_weights()) {
    if (w == nullptr) {
      LBANN_ERROR("callback \"", name(), "\" "
                  "got a weights pointer that is a null pointer");
    }
    if(w->get_name() != m_output_name) {
      model_size += w->get_size();
    }
  }
  if (m.get_comm()->am_trainer_master()) {
    std::cout << "Trainer [ " << m.get_comm()->get_trainer_rank() << " ], Step " << m.get_execution_context().get_step();
    std::cout << " Model size  " <<  model_size << std::endl;
  }

  auto& out_w = dynamic_cast<data_type_weights<DataType>&>(*m_output);
  out_w.set_value(model_size,0);
  
}


std::unique_ptr<callback_base>
build_compute_model_size_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackComputeModelSize&>(proto_msg);
  return std::make_unique<compute_model_size>(
    params.output_name(),
    params.batch_interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::compute_model_size
#include <lbann/macros/register_class_with_cereal.hpp>
