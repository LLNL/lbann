////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include <cereal/access.hpp>
#include <memory>
#include <vector>

namespace lbann {

/** @brief Layer composed of one or more operator objects
 *
 *  Operators are applied sequentially.
 */
template <typename InputT, typename OutputT, data_layout Layout, El::Device D>
class OperatorLayer final : public data_type_layer<InputT, OutputT>
{
  using DataTypeLayer = data_type_layer<InputT, OutputT>;
  using OperatorType = Operator<InputT, OutputT, D>;
  using OperatorPtr = std::unique_ptr<OperatorType>;

  std::vector<OperatorPtr> m_ops;

public:
  /** @name Lifecycle functions */
  ///@{
  /** @brief Construct from a single operator. */
  OperatorLayer(lbann_comm& comm, OperatorPtr op);
  /** @brief Construct from a vector of operators. */
  OperatorLayer(lbann_comm& comm, std::vector<OperatorPtr> operators);

  /** @brief Copy constructor. */
  OperatorLayer(OperatorLayer const& other);
  /** @brief Copy assignment. */
  OperatorLayer& operator=(OperatorLayer const& other);

  /** @brief Move constructor. */
  OperatorLayer(OperatorLayer&& other) = default;
  /** @brief Move assignment. */
  OperatorLayer& operator=(OperatorLayer&& other) = default;

  /** @brief Destructor. */
  ~OperatorLayer() = default;

  /** @brief Polymorphic copy. */
  OperatorLayer* copy() const final;
  ///@}

  std::string get_type() const final;
  data_layout get_data_layout() const final;
  El::Device get_device_allocation() const final;
  bool can_run_inplace() const final;
  int get_backprop_requirements() const final;

  void fp_compute() final;
  void bp_compute() final;

  description get_description() const final;

  template <typename ArchiveT>
  void serialize(ArchiveT&);

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  friend cereal::access;
  OperatorLayer();

  static std::vector<OperatorPtr>
  clone_ops(std::vector<OperatorPtr> const& ops);

  static std::vector<size_t> fix_type(std::vector<int> const& in);

  std::vector<utils::ConstDistTensorView<InputT, D>> get_inputs() const;
  std::vector<utils::DistTensorView<OutputT, D>> get_outputs();
  std::vector<utils::ConstDistTensorView<OutputT, D>>
  get_grad_wrt_outputs() const;
  std::vector<utils::DistTensorView<InputT, D>> get_grad_wrt_inputs();

}; // class OperatorLayer

template <typename InputT,
          typename OutputT,
          data_layout Layout,
          El ::Device Device>
std::unique_ptr<Layer> build_operator_layer_from_pbuf(lbann_comm*,
                                                      lbann_data::Layer const&);

} // namespace lbann
#endif // LBANN_LAYERS_OPERATOR_LAYER_HPP_INCLUDED
