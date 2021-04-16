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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/comm_impl.hpp"
#include "lbann/callbacks/compute_model_size.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/serialize.hpp"

#include <callbacks.pb.h>


namespace lbann {
namespace callback {

compute_model_size::compute_model_size(std::string output_layer_name,
                           El::Int batch_interval)
  : callback_base(batch_interval),
    m_output_layer_name(std::move(output_layer_name)) {}

compute_model_size::compute_model_size()
  : compute_model_size("",0)
{}

template <class Archive>
void compute_model_size::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_output_layer_name));
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
constant_layer<TensorDataType, T_layout, Dev>* compute_model_size::get_constant_layer(Layer* l) {
  if(auto c_layer = dynamic_cast<constant_layer<TensorDataType, T_layout, Dev>*>(l)) return c_layer;
  else return nullptr;
}

void compute_model_size::setup(model* m) {
   for(auto* l : m->get_layers()) {
      if(l->get_name() == m_output_layer_name){
        m_output_layer = l;
        break;
      }
   }
    if (m_output_layer->get_type() != "constant") {
      LBANN_ERROR("Current implementation of callback \"", name(), "\" "
                  "assumes constant layer as output");
    }
}

void compute_model_size::on_batch_begin(model* m) {
  const auto& c = m->get_execution_context();
  if (m_output_layer != nullptr && 
      c.get_step() % m_batch_interval == 0 &&
      c.get_step() > 0) {
    compute_size(*m);
  }
}

void compute_model_size::compute_size(model& m){
  DataType model_size = 0; //size_t
  for (auto* w : m.get_weights()) {
    if (w == nullptr) {
      LBANN_ERROR("callback \"", name(), "\" "
                  "got a weights pointer that is a null pointer");
    }
    model_size += w->get_size();
  }
  //print per trainer model size
  if (m.get_comm()->am_trainer_master()) {
    std::cout << "Trainer [ " << m.get_comm()->get_trainer_rank() << " ], Iteration " << m_batch_interval;
    std::cout << "  Model size  " << model_size << std::endl;
  }
  auto cpu_dp_l = get_constant_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>(m_output_layer);
  if(cpu_dp_l) El::Fill(cpu_dp_l->get_activations(), model_size);
  auto cpu_mp_l = get_constant_layer<DataType, data_layout::MODEL_PARALLEL, El::Device::CPU>(m_output_layer);
  if(cpu_mp_l) El::Fill(cpu_mp_l->get_activations(), model_size);
  #ifdef LBANN_HAS_GPU
  auto gpu_dp_l = get_constant_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>(m_output_layer);
  if(gpu_dp_l) El::Fill(gpu_dp_l->get_activations(), model_size);
  auto gpu_mp_l = get_constant_layer<DataType, data_layout::MODEL_PARALLEL, El::Device::GPU>(m_output_layer);
  if(gpu_mp_l) El::Fill(gpu_mp_l->get_activations(), model_size);
  #endif

}


std::unique_ptr<callback_base>
build_compute_model_size_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackComputeModelSize&>(proto_msg);
  return make_unique<compute_model_size>(
    params.output_layer_name(),
    params.batch_interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::compute_model_size
#include <lbann/macros/register_class_with_cereal.hpp>
