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

#include "lbann/base.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/exception.hpp"
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
  using OperatorPtr = std::unique_ptr<Operator<InputT, OutputT, D>>;

private:
  std::vector<OperatorPtr> m_ops;

public:
  OperatorLayer(lbann_comm& comm, OperatorPtr op)
    : DataTypeLayer(&comm)
  {
    m_ops.reserve(1);
    m_ops.emplace_back(std::move(op));
  }
  OperatorLayer(lbann_comm& comm, std::vector<OperatorPtr> operators)
    : DataTypeLayer(&comm), m_ops{std::move(operators)}
  {
    LBANN_ASSERT(m_ops.size() == 1UL); // For starters.
  }
  OperatorLayer(OperatorLayer const& other)
    : DataTypeLayer(other), m_ops{clone_ops(other.m_ops)}
  {}

  ~OperatorLayer() = default;

  OperatorLayer* copy() const override
  {
    return new OperatorLayer<InputT, OutputT, Layout, D>(*this);
  }
  std::string get_type() const override { return "operator"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return D; }

  void fp_compute() override
  {
    return m_ops[0]->fp_compute(this->get_inputs(), this->get_outputs());
  }

  void bp_compute() override
  {
    return m_ops[0]->bp_compute(this->get_inputs(),
                                this->get_grad_wrt_outputs(),
                                this->get_grad_wrt_inputs());
  }

  description get_description() const override
  {
    auto desc = DataTypeLayer::get_description();
    for (auto const& op : m_ops)
      desc.add(op->get_description());
    return desc;
  }

private:
  static auto clone_ops(std::vector<OperatorPtr> const& ops)
  {
    std::vector<OperatorPtr> out;
    out.reserve(ops.size());
    for (auto const& x : ops) {
      out.emplace_back(x->clone());
    }
    return out;
  }

  static auto fix_type(std::vector<int> const& in)
  {
    return std::vector<size_t>{cbegin(in), cend(in)};
  }

  std::vector<utils::ConstDistTensorView<InputT, D>> get_inputs() const
  {
    auto n_parents = this->get_num_parents();
    std::vector<utils::ConstDistTensorView<InputT, D>> out;
    out.reserve(n_parents);
    for (int p = 0; p < n_parents; ++p)
      out.emplace_back(this->get_prev_activations(p),
                       fix_type(this->get_input_dims(p)));
    return out;
  }
  std::vector<utils::DistTensorView<OutputT, D>> get_outputs()
  {
    auto n_children = this->get_num_children();
    std::vector<utils::DistTensorView<OutputT, D>> out;
    out.reserve(n_children);
    for (int c = 0; c < n_children; ++c)
      out.emplace_back(this->get_activations(c),
                       fix_type(this->get_output_dims(c)));
    return out;
  }

  std::vector<utils::ConstDistTensorView<OutputT, D>>
  get_grad_wrt_outputs() const
  {
    auto n_children = this->get_num_children();
    std::vector<utils::ConstDistTensorView<OutputT, D>> out;
    out.reserve(n_children);
    for (int c = 0; c < n_children; ++c)
      out.emplace_back(this->get_prev_error_signals(c),
                       fix_type(this->get_output_dims(c)));
    return out;
  }

  std::vector<utils::DistTensorView<InputT, D>> get_grad_wrt_inputs()
  {
    auto n_parents = this->get_num_parents();
    std::vector<utils::DistTensorView<InputT, D>> out;
    out.reserve(n_parents);
    for (int p = 0; p < n_parents; ++p)
      out.emplace_back(this->get_error_signals(p),
                       fix_type(this->get_input_dims(p)));
    return out;
  }

}; // class OperatorLayer

} // namespace lbann
#endif // LBANN_LAYERS_OPERATOR_LAYER_HPP_INCLUDED
