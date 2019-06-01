////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/callback_perturb_dropout.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

lbann_callback_perturb_dropout::lbann_callback_perturb_dropout(EvalType keep_prob_factor,
                                                         std::set<std::string> layer_names)
  : lbann_callback(1),
    m_keep_prob_factor(keep_prob_factor),
    m_layer_names(std::move(layer_names)) {}

void lbann_callback_perturb_dropout::setup(model* m) {
  perturb(*m);
}

template <data_layout T_layout, El::Device Dev>
dropout<T_layout, Dev>* lbann_callback_perturb_dropout::get_dropout_layer(Layer* l) {
  if(auto d_layer = dynamic_cast<dropout<T_layout, Dev>*>(l)) return d_layer;
  else return nullptr;
}

void lbann_callback_perturb_dropout::perturb(model& m) {
  auto* comm = m.get_comm();
  for (auto* l : m.get_layers()) {
    if (l == nullptr) {
      std::stringstream err;
      err << "callback \"" << name() << "\" "
          << "got a layer pointer that is a null pointer";
      LBANN_ERROR(err.str());
    }
    if (m_layer_names.empty()
        || m_layer_names.count(l->get_name()) > 0) {
      
      auto d_dp_cpu = get_dropout_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(l);
      auto d_mp_cpu = get_dropout_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(l);
      #ifdef LBANN_HAS_GPU
      auto d_dp_gpu = get_dropout_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(l);
      auto d_mp_gpu = get_dropout_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(l);
      #endif
      // Perturb dropout layer
        if(d_dp_cpu != nullptr || d_mp_cpu != nullptr 
           #ifdef LBANN_HAS_GPU
           || d_dp_gpu != nullptr || d_mp_gpu != nullptr
           #endif
          ) {
        EvalType new_keep_prob;
        if (comm->am_trainer_master()) {

          // Useful constants
          constexpr EvalType zero = 0;
          constexpr EvalType one = 1;
          constexpr EvalType min_val = std::numeric_limits<EvalType>::min();

          // RNG
          auto& gen = get_generator();
          std::normal_distribution<EvalType> dist(zero, one);

          // Perturb log(keep_prob)
          EvalType old_keep_prob = 0;
          if (d_dp_cpu) old_keep_prob = d_dp_cpu->get_keep_prob();
          if (d_mp_cpu) old_keep_prob = d_mp_cpu->get_keep_prob();
          #ifdef LBANN_HAS_GPU
          if (d_dp_gpu) old_keep_prob = d_dp_gpu->get_keep_prob();
          if (d_mp_gpu) old_keep_prob = d_mp_gpu->get_keep_prob();
          #endif
          if (m_keep_prob_factor > zero) {
            auto log_val = std::log(one - std::max(old_keep_prob, min_val));
            log_val += m_keep_prob_factor * dist(gen);
            new_keep_prob = std::max(EvalType(0.5), std::min(one - std::exp(log_val),one));
            std::cout << "Trainer [ " << comm->get_trainer_rank() << " ] keep prob changed from "
                << old_keep_prob << " to " << new_keep_prob << std::endl;
          }

        }

        // Communicate new keep prob from trainer master processes
        comm->trainer_broadcast(comm->get_trainer_master(), new_keep_prob);

        // Update keep prob
        if (d_dp_cpu) d_dp_cpu->set_keep_prob(new_keep_prob);
        if (d_mp_cpu) d_mp_cpu->set_keep_prob(new_keep_prob);
        #ifdef LBANN_HAS_GPU
        if (d_dp_gpu) d_dp_gpu->set_keep_prob(new_keep_prob);
        if (d_mp_gpu) d_mp_gpu->set_keep_prob(new_keep_prob);
        #endif

      }

    }
  }
}


} // namespace lbann
