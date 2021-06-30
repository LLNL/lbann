////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_LAYERS_OPERATOR_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_OPERATOR_LAYER_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/utils/describable.hpp"
#include "lbann/utils/tensor.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace lbann {

template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
class OperatorLayer : public data_type_layer<InputT, OutputT>
{
private:
  using DataTypeLayer = data_type_layer<InputT, OutputT>;
  using OperatorType = Operator<InputT, OutputT, D>;
  using OperatorPtr = std::unique_ptr<OperatorType>;

private:
  std::vector<OperatorPtr> m_ops;

public:
  OperatorLayer(lbann_comm& comm, OperatorPtr op);
  OperatorLayer(lbann_comm& comm, std::vector<OperatorPtr> operators);
  OperatorLayer(OperatorLayer const& other);

  ~OperatorLayer() = default;

  OperatorLayer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  void fp_compute() override;
  void bp_compute() override;

  description get_description() const override;

private:
  static std::vector<OperatorPtr>
  clone_ops(std::vector<OperatorPtr> const& ops);

  static std::vector<size_t> fix_type(std::vector<int> const& in);

  std::vector<utils::ConstDistTensorView<InputT, D>> get_inputs() const;
  std::vector<utils::DistTensorView<OutputT, D>> get_outputs();
  std::vector<utils::ConstDistTensorView<OutputT, D>>
  get_grad_wrt_outputs() const;
  std::vector<utils::DistTensorView<InputT, D>> get_grad_wrt_inputs();

}; // class OperatorLayer

LBANN_DEFINE_LAYER_BUILDER(operator);

} // namespace lbann
#endif // LBANN_LAYERS_OPERATOR_LAYER_HPP_INCLUDED
