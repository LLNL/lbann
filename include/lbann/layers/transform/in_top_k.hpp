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

#ifndef LBANN_LAYER_IN_TOP_K_HPP_INCLUDED
#define LBANN_LAYER_IN_TOP_K_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Indicate top-k entries.
 *
 *  Output entries corresponding to the top-k input entries are set to
 *  one and the rest to zero. Ties are broken in favor of entries with
 *  smaller indices.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class in_top_k_layer : public transform_layer {
 public:

  in_top_k_layer(lbann_comm *comm, El::Int k)
    : transform_layer(comm), m_k(k) {
    if (m_k < 0) {
      std::stringstream err;
      err << "invalid parameter for top-k search (k=" << m_k << ")";
      LBANN_ERROR(err.str());
    }
  }

  in_top_k_layer* copy() const override { return new in_top_k_layer(*this); }
  std::string get_type() const override { return "in_top_k"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    desc.add("k", m_k);
    return desc;
  }

 protected:

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims(get_input_dims());
  }

  void fp_compute() override;

 private:

  /** Parameter for top-k search. */
  const El::Int m_k;

};

} // namespace lbann

#endif // LBANN_LAYER_IN_TOP_K_HPP_INCLUDED
