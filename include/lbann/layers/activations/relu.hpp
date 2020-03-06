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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

/** Rectified linear unit activation function layer.
 *  \f[ ReLU(x) = \text{max}(x, 0) \f]
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class relu_layer : public data_type_layer<TensorDataType> {
public:
  relu_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {}
  relu_layer* copy() const override { return new relu_layer(*this); }
  std::string get_type() const override { return "ReLU"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:
  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
 protected:
  dc::ReLU *m_relu;
  void fp_compute_distconv();
  void bp_compute_distconv();

 public:
  void init_distribution(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override;
  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override;
  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override;

#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
