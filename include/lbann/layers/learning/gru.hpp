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

#ifndef LBANN_LAYERS_LEARNING_GRU_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_GRU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB
#ifdef LBANN_HAS_ONEDNN
#include "lbann/utils/dnn_lib/onednn.hpp"
#endif // LBANN_HAS_ONEDNN

// Supported implementations -- See lbann_config.h

namespace lbann {

/** @brief Stacked gated recurrent unit
 *
 *  Expects two inputs: a 2D input sequence (
 *  @f$ \text{sequence\_length}\times\text{input\_size} @f$ )
 *  and a 2D initial hidden state (
 *  @f$ \text{num\_layers}times\text{hidden\_size} @f$ ).
 *
 *  Uses four weights per GRU cell: "ih\_matrix" (
 *  @f$ 3 \text{hidden\_size}\times\text{input\_size} @f$ for layer 0
 *  and @f$ 3 \text{hidden\_size}\times\text{hidden\_size} @f$ for other
 *  layers), "hh\_matrix" (
 *  @f$ 3 \text{hidden\_size}\times\text{hidden\_size} @f$ ),
 *  "ih_bias" ( @f$ 3 \text{hidden\_size} @f$ ),
 *  "hh_bias" ( @f$ 3 \text{hidden\_size} @f$ ).
 *
 *  Support is experimental and requires either cuDNN (on GPU) or
 *  oneDNN (on CPU).
 *
 *  @todo Support bidirectional RNNs
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class gru_layer : public data_type_layer<TensorDataType>
{

  static_assert(Layout == data_layout::DATA_PARALLEL,
                "GRU layer only supports data parallel layout");

public:
  gru_layer(size_t hidden_size, size_t num_layers);

  gru_layer(const gru_layer& other);
  gru_layer& operator=(const gru_layer& other);
  ~gru_layer() = default;

  gru_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS;
  }

  description get_description() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  size_t get_hidden_size() const { return m_hidden_size; }
  size_t get_num_layers() const { return m_num_layers; }
  const hydrogen::simple_buffer<El::byte, Device>& get_reserve_space() const;

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  gru_layer() : gru_layer(0, 0) {}

  void setup_dims() override;
  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  /** @brief Size of each hidden state and output vector */
  size_t m_hidden_size;
  /** @brief Number of stacked GRU cells */
  size_t m_num_layers;

#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
  /** @name oneDNN CPU implementation */
  ///@{

  /** @brief Objects used in oneDNN CPU implementation */
  struct OnednnCpuObjects
  {

    // Typedefs
    using Backend = onednn_backend<El::Device::CPU>;
    using TensorDesc = Backend::TensorDescriptor;

    // Descriptors
    ::dnnl::lbr_gru_forward::primitive_desc gru_forward_primitive_desc;
    ::dnnl::lbr_gru_forward::primitive gru_forward_primitive;
    ::dnnl::lbr_gru_backward::primitive_desc gru_backward_primitive_desc;
    ::dnnl::lbr_gru_backward::primitive gru_backward_primitive;
    TensorDesc input_sequence_desc;
    TensorDesc init_hidden_desc;
    TensorDesc output_sequence_desc;
    TensorDesc final_hidden_desc;
    TensorDesc input_sequence_grad_desc;
    TensorDesc init_hidden_grad_desc;
    TensorDesc output_sequence_grad_desc;
    TensorDesc final_hidden_grad_desc;

    // Workspaces
    TensorDesc forward_ih_matrix_weights;
    TensorDesc forward_hh_matrix_weights;
    TensorDesc backward_ih_matrix_weights;
    TensorDesc backward_hh_matrix_weights;
    TensorDesc bias_weights;
    TensorDesc ih_matrix_weights_grad;
    TensorDesc hh_matrix_weights_grad;
    TensorDesc bias_weights_grad;
    TensorDesc workspace;
  };

  /** @brief Storage for oneDNN CPU objects */
  std::unique_ptr<OnednnCpuObjects> m_onednn_cpu_objects;

  /** @brief Setup oneDNN CPU implementation */
  void setup_onednn_cpu();

  ///@}
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED

#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  /** @name cuDNN implementation */
  ///@{

  /** @brief Objects used in cuDNN implementation */
  struct CudnnObjects
  {

    // Typedefs
    using ByteBuffer = hydrogen::simple_buffer<El::byte, El::Device::GPU>;
    using IntBuffer = hydrogen::simple_buffer<int32_t, El::Device::GPU>;
    using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
    using GraphCache =
      std::unordered_map<size_t, std::pair<size_t, cuda::ExecutableGraph>>;

    // Descriptors
    dnn_lib::RNNDescriptor rnn_desc;
    dnn_lib::RNNDataDescriptor input_desc;
    dnn_lib::RNNDataDescriptor output_desc;
    dnn_lib::TensorDescriptor hidden_desc;

    // Workspaces
    LocalMat input_sequence_workspace;
    LocalMat output_sequence_workspace;
    LocalMat input_sequence_grad_workspace;
    LocalMat output_sequence_grad_workspace;
    LocalMat init_hidden_workspace;
    LocalMat init_hidden_grad_workspace;
    ByteBuffer weights_workspace;
    ByteBuffer weights_grad_workspace;
    ByteBuffer workspace;
    ByteBuffer reserve_space;
    IntBuffer gpu_sequence_lengths;

    /** The cache is a map from mini-batch sizes to (hash, graph)
     *  pairs. The hash is generated from the cuDNN function
     *  arguments, mostly pointers. The graph is a @c
     *  cuda::ExecutableGraph .
     */
    GraphCache forward_prop_graph_cache;
    /** The cache is a map from mini-batch sizes to (hash, graph)
     *  pairs. The hash is generated from the cuDNN function
     *  arguments, mostly pointers. The graph is a @c
     *  cuda::ExecutableGraph .
     */
    GraphCache backward_prop_graph_cache;
  };

  /** @brief Storage for cuDNN objects */
  std::unique_ptr<CudnnObjects> m_cudnn_objects;

  /** @brief Setup cuDNN implementation */
  void setup_cudnn();

  ///@}
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED

  template <typename T>
  friend void fp_compute_impl(gru_layer<T, Layout, Device>&);
  template <typename T>
  friend void bp_compute_impl(gru_layer<T, Layout, Device>&);
};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(gru);

// Explicit template instantiation
#ifndef LBANN_GRU_LAYER_INSTANTIATE

#ifdef LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
#define PROTO(T)                                                               \
  extern template class gru_layer<T,                                           \
                                  data_layout::DATA_PARALLEL,                  \
                                  El::Device::CPU>;
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED

#ifdef LBANN_GRU_LAYER_CUDNN_SUPPORTED
#define PROTO(T)                                                               \
  extern template class gru_layer<T,                                           \
                                  data_layout::DATA_PARALLEL,                  \
                                  El::Device::GPU>;
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED

#endif // LBANN_GRU_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_GRU_HPP_INCLUDED
