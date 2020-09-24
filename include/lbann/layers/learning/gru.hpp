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

#ifndef LBANN_LAYERS_LEARNING_GRU_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_GRU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/cudnn.hpp"
#endif // LBANN_HAS_CUDNN

namespace lbann {

/** @brief Gated recurrent unit
 *
 *  Expects two inputs: a 2D input sequence (
 *  @f$ \text{sequence\_length}\times\text{input\_size} @f$ )
 *  and a 1D initial hidden state ( @f$ \text{hidden\_size} @f$ ).
 *
 *  Uses four weights: "ih\_matrix" (
 *  @f$ 3 \text{hidden\_size}\times\text{input\_size} @f$ ),
 *  "hh\_matrix" (
 *  @f$ 3 \text{hidden\_size}\times\text{hidden\_size} @f$ ),
 *  "ih_bias" ( @f$ 3 \text{hidden\_size} @f$ ),
 *  "hh_bias" ( @f$ 3 \text{hidden\_size} @f$ ).
 *
 *  @todo Support CPU
 *  @todo Support bidirectional RNNs
 *  @todo Support stacked RNNs
 *
 *  @warning cuDNN 8 exposes a new RNN API and deprecates the old one.
 *  Consider reimplementing this layer once cuDNN 8 is the minimum
 *  version.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class gru_layer
  : public data_type_layer<TensorDataType> {

  static_assert(Layout == data_layout::DATA_PARALLEL,
                "GRU layer only supports data parallel layout");

public:

  gru_layer(
    lbann_comm* comm,
    size_t hidden_size);

  gru_layer(const gru_layer& other);
  gru_layer& operator=(const gru_layer& other);
  ~gru_layer() = default;

  gru_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  description get_description() const override;

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void setup_data(size_t max_mini_batch_size) override;
#ifdef LBANN_HAS_CUDNN
  void setup_gpu() override;
#endif // LBANN_HAS_CUDNN

  void fp_compute() override;
  void bp_compute() override;

private:

  /** @brief  Size of each hidden state and output vector */
  size_t m_hidden_size;

#ifdef LBANN_HAS_CUDNN
  using ByteBuffer = hydrogen::simple_buffer<El::byte, Device>;
  cudnn::RNNDescriptor m_rnn_cudnn_desc;
  cudnn::DropoutDescriptor m_dropout_cudnn_desc;
  cudnn::RNNDataDescriptor m_input_cudnn_desc;
  cudnn::RNNDataDescriptor m_output_cudnn_desc;
  cudnn::TensorDescriptor m_hidden_cudnn_desc;
  cudnn::FilterDescriptor m_packed_weights_cudnn_desc;
  ByteBuffer m_cudnn_reserve_space;
  hydrogen::simple_buffer<int32_t, El::Device::GPU> m_gpu_sequence_lengths;
  cuda::ExecutableGraph m_graph_forward_prop;
  cuda::ExecutableGraph m_graph_backward_prop;
#endif // LBANN_HAS_CUDNN

  template <typename T>
  friend void fp_compute_impl(gru_layer<T,Layout,Device>&);
  template <typename T>
  friend void bp_compute_impl(gru_layer<T,Layout,Device>&);

};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(gru);

// Explicit template instantiation
#ifdef LBANN_HAS_CUDNN
#ifndef LBANN_GRU_LAYER_INSTANTIATE
#define PROTO(T)                                                        \
  extern template class gru_layer<                                             \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_GRU_LAYER_INSTANTIATE
#endif // LBANN_HAS_CUDNN

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_GRU_HPP_INCLUDED
