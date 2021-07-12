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

#ifndef LBANN_LAYER_EVALUATION_HPP_INCLUDED
#define LBANN_LAYER_EVALUATION_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Interface with objective function and metrics. */
template <typename TensorDataType>
class abstract_evaluation_layer : public data_type_layer<TensorDataType> {
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
  /** Get evaluated value. */
  EvalType get_value(bool scaled = true);

  /** Construct an evaluation layer.
   *  The caller is responsible for deallocating the layer.
   */
  static abstract_evaluation_layer* construct(lbann_comm *comm,
                                              data_layout layout,
                                              El::Device device);

protected:
  abstract_evaluation_layer(lbann_comm *comm);
  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  friend class cereal::access;
  abstract_evaluation_layer()
    : abstract_evaluation_layer(nullptr)
  {}

  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void setup_data(size_t max_mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Scaling factor to apply to evaluated value. */
  EvalType m_scale = 0;
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
class evaluation_layer : public abstract_evaluation_layer<TensorDataType> {
public:
  evaluation_layer(lbann_comm *comm) : abstract_evaluation_layer<TensorDataType>(comm) {}
  evaluation_layer* copy() const override { return new evaluation_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "evaluation"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  std::string get_onnx_op_type() const override { return "Identity"; }

  void fill_onnx_node(onnx::GraphProto& graph) const override {
    auto* node = graph.add_node();
    for(auto const* parent : this->get_parent_layers()) {
      size_t idx = parent->find_child_layer_index(*this);
      node->add_input(parent->get_name() + "_" + std::to_string(idx));
    }
    node->add_output(this->get_name());
    node->set_name(this->get_name());
    node->set_op_type(this->get_onnx_op_type());
    node->set_domain("");
    node->set_doc_string(this->get_type());

    // Add graph output
    auto graph_output = graph.add_output();
    graph_output->set_name(this->get_name());
    auto* graph_output_type = graph_output->mutable_type();
    // FIXME: enum type. 1 is float
    graph_output_type->mutable_tensor_type()->set_elem_type(1);

    auto* dims = graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
    dims->set_dim_param("batch");
    dims = graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
    dims->set_dim_value(1);
    dims = graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
    dims->set_dim_value(28);
    dims = graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
    dims->set_dim_value(28);

    //initializer->add_dims(1);
    //auto const parents = this->get_parent_layers();
    //for( auto const& dim : parents[0]->get_output_dims() ) {
      //initializer->add_dims(dim);
      //dims = graph_output_type->mutable_tensor_type()->mutable_shape()->add_dim();
      //dims->set_dim_value(dim);
      //dims->set_denotation("N/A");
    // }

  }

protected:
  friend class cereal::access;
  evaluation_layer()
    : evaluation_layer(nullptr)
  {}

};

LBANN_DEFINE_LAYER_BUILDER(evaluation);

#ifndef LBANN_EVALUATION_LAYER_INSTANTIATE
#define PROTO(T)                           \
  extern template class abstract_evaluation_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#define PROTO_DEVICE(T, Device)                                         \
  extern template class evaluation_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class evaluation_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_EVALUATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_EVALUATION_HPP_INCLUDED
