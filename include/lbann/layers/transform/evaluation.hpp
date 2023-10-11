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

#ifndef LBANN_LAYER_EVALUATION_HPP_INCLUDED
#define LBANN_LAYER_EVALUATION_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Interface with objective function and metrics */
template <typename TensorDataType>
class abstract_evaluation_layer : public data_type_layer<TensorDataType>
{
public:
#ifdef LBANN_DETERMINISTIC
  using EvalDataType = EvalType;
#else
  using EvalDataType = TensorDataType;
#endif
  using CPUMatType = El::Matrix<EvalDataType, El::Device::CPU>;

public:
  /** Get scaling factor. */
  EvalType get_scale() const { return m_scale; }
  /** Set scaling factor. */
  void set_scale(EvalType scale) { m_scale = scale; }
  /** Set the AMP scaling factor. */
  void set_amp_scale(EvalType scale) { m_amp_scale = scale; /*std::cout << "Set AMP scale to " << scale << std::endl;*/ }
  /** Get evaluated value. */
  EvalType get_value(bool scaled = true);

  /** Construct an evaluation layer.
   *  The caller is responsible for deallocating the layer.
   */
  static abstract_evaluation_layer*
  construct(lbann_comm* comm, data_layout layout, El::Device device);

protected:
  abstract_evaluation_layer(lbann_comm* comm);
  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  friend class cereal::access;
  abstract_evaluation_layer() : abstract_evaluation_layer(nullptr) {}

  void setup_dims() override;
  void setup_data(size_t max_mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:
  /** Scaling factor to apply to evaluated value. */
  EvalType m_scale = 0;
  /** Scaling factor for automatic mixed precision.
   *  This is used only to scale the backpropagated loss value.
   */
  EvalType m_amp_scale = 0;
  /** Evaluated value.
   *  The value may be stored in pinned memory.
   */
  CPUMatType m_value;
  /** Non-blocking allreduce request. */
  Al::request m_allreduce_req;
#ifdef LBANN_HAS_GPU
  /** CUDA event after a non-blocking GPU-CPU memory copy. */
  gpu_lib::event_wrapper m_copy_event;
#endif // LBANN_HAS_GPU
};

/** Evaluation layer.
 *  Computes the average value across a mini-batch. If the input
 *  tensor has multiple neurons, their values are added together.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class evaluation_layer : public abstract_evaluation_layer<TensorDataType>
{
public:
  evaluation_layer(lbann_comm* comm)
    : abstract_evaluation_layer<TensorDataType>(comm)
  {}
  evaluation_layer* copy() const override
  {
    return new evaluation_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "evaluation"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return PREV_ACTIVATIONS | ERROR_SIGNALS;
  }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  evaluation_layer() : evaluation_layer(nullptr) {}
};

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void evaluation_layer<T, L, D>::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto* eval = graph.add_node();
  for (auto const* parent : this->get_parent_layers()) {
    size_t idx = parent->find_child_layer_index(*this);
    eval->add_input(parent->get_name() + "_" + std::to_string(idx));
  }
  eval->add_output(this->get_name());
  eval->set_name(this->get_name());
  eval->set_op_type("Identity");
  eval->set_domain("");
  eval->set_doc_string(this->get_type());

  // Add graph output
  auto graph_output = graph.add_output();
  graph_output->set_name(eval->output(0));
  auto* graph_output_type = graph_output->mutable_type();
  graph_output_type->mutable_tensor_type()->set_elem_type(
    onnx::AttributeProto::FLOAT);

  auto* dims =
    graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
  dims->set_dim_param("batch");
  dims = graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
  dims->set_dim_value(1);
}
#endif // LBANN_HAS_ONNX

#ifndef LBANN_EVALUATION_LAYER_INSTANTIATE
#define PROTO(T) extern template class abstract_evaluation_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#define PROTO_DEVICE(T, Device)                                                \
  extern template class evaluation_layer<T,                                    \
                                         data_layout::DATA_PARALLEL,           \
                                         Device>;                              \
  extern template class evaluation_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_EVALUATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_EVALUATION_HPP_INCLUDED
