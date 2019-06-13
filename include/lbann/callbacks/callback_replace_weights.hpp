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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_REPLACE_WEIGHTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_REPLACE_WEIGHTS_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 *  Weights/parameters replacement on k-batch end
 *  Currently support replacing weights/parameters using layer names
 *  Can easily be extended to support replacement by weights name
 *  Given two layers specified in prototext, weights are copied from source layer to destination layer.
 */
class lbann_callback_replace_weights : public lbann_callback {
 public:
  lbann_callback_replace_weights(std::vector<Layer*> src,
    std::vector<Layer*> dst, int batch_interval=1) :
    lbann_callback(batch_interval),
    m_src_layers(std::move(src)),
    m_dst_layers(std::move(dst)){
    if(m_src_layers.size() != m_dst_layers.size())
     throw lbann_exception("In replace weights callback: number of src and dest layers does not match.");
  }

  lbann_callback_replace_weights(
    const lbann_callback_replace_weights&) = default;
  lbann_callback_replace_weights& operator=(
    const lbann_callback_replace_weights&) = default;
  lbann_callback_replace_weights* copy() const override {
    return new lbann_callback_replace_weights(*this);
  }
  void on_batch_end(model *m) override;

  std::string name() const override { return "replace weights"; }
 private:
  std::vector<Layer*> m_src_layers, m_dst_layers;

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_REPLACE_WEIGHTS_HPP_INCLUDED
