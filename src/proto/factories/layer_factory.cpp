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

#include "lbann/proto/factories.hpp"

namespace lbann {
namespace proto {
  
template <data_layout layout>
Layer* construct_layer(lbann_comm* comm,
                       std::map<execution_mode, generic_data_reader*>& data_readers,
                       int num_parallel_readers,
                       cudnn::cudnn_manager* cudnn,
                       const lbann_data::Layer& proto_layer) {
  std::stringstream err;

  // Currently only data-parallel layers have GPU support
  /// @todo Support for GPU model-parallel layers
  if (layout == data_layout::MODEL_PARALLEL) { cudnn = nullptr; }

  // Input layers
  if (proto_layer.has_input()) {
    const auto& proto_input = proto_layer.input();
    const auto& io_buffer = proto_input.io_buffer();
    const auto& data_set_per_model = proto_input.data_set_per_model();
    const auto& for_regression = proto_input.for_regression();
    if (io_buffer == "distributed") {
      return new input_layer<distributed_io_buffer, layout>(comm,
                                                            num_parallel_readers,
                                                            data_readers,
                                                            !data_set_per_model,
                                                            for_regression);
    }
    if (io_buffer == "partitioned") {
      return new input_layer<partitioned_io_buffer, layout>(comm,
                                                            num_parallel_readers,
                                                            data_readers,
                                                            !data_set_per_model,
                                                            for_regression);
    }
  }

  // Target layers
  if (proto_layer.has_target()) {
    const auto& proto_target = proto_layer.target();
    const auto& io_buffer = proto_target.io_buffer();
    const auto& shared_data_reader = proto_target.shared_data_reader();
    const auto& for_regression = proto_target.for_regression();
    if (io_buffer == "distributed") {
      return new target_layer<distributed_io_buffer, layout>(comm,
                                                             nullptr,
                                                             num_parallel_readers,
                                                             data_readers,
                                                             shared_data_reader,
                                                             for_regression);
    }
    if (io_buffer == "partitioned") {
      return new target_layer<partitioned_io_buffer, layout>(comm,
                                                             nullptr,
                                                             num_parallel_readers,
                                                             data_readers,
                                                             shared_data_reader,
                                                             for_regression);
    }
  }
  if (proto_layer.has_reconstruction()) {
    return new reconstruction_layer<layout>(comm, nullptr);
  }

  // Fully connected layer
  if (proto_layer.has_fully_connected()) {
    const auto& proto_fc = proto_layer.fully_connected();
    int num_neurons = proto_fc.num_neurons();
    const auto& bias = proto_fc.has_bias();
    if (proto_layer.num_neurons_from_data_reader()) {
      num_neurons = data_readers[execution_mode::training]->get_linearized_data_size();
    }
    return new fully_connected_layer<layout>(comm,
                                             num_neurons,
                                             nullptr,
                                             bias,
                                             cudnn);
  }

  // Convolution and deconvolution layer
  if (proto_layer.has_convolution()) {
    const auto& proto_conv = proto_layer.convolution();
    const auto& num_output_channels = proto_conv.num_output_channels();
    const auto& bias = proto_conv.has_bias();
    if (proto_conv.has_vectors()) {
      const auto& dims = parse_list<int>(proto_conv.conv_dims());
      const auto& pads = parse_list<int>(proto_conv.conv_pads());
      const auto& strides = parse_list<int>(proto_conv.conv_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, bias, cudnn
                   );
      }
    } else {
      const auto& num_dims = proto_conv.num_dims();
      const auto& dim = proto_conv.conv_dims_i();
      const auto& pad = proto_conv.conv_pads_i();
      const auto& stride = proto_conv.conv_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, bias, cudnn
                   );
      }
    }
  }
  if (proto_layer.has_deconvolution()) {
    const auto& proto_deconv = proto_layer.deconvolution();
    const auto& bias = proto_deconv.has_bias();
    int num_output_channels = proto_deconv.num_output_channels();
    if (proto_layer.num_neurons_from_data_reader()) {
      num_output_channels = data_readers[execution_mode::training]->get_linearized_data_size();
    }
    if (proto_deconv.has_vectors()) {
      const auto& dims = parse_list<int>(proto_deconv.conv_dims());
      const auto& pads = parse_list<int>(proto_deconv.conv_pads());
      const auto& strides = parse_list<int>(proto_deconv.conv_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, bias, cudnn
                   );
      }
    } else {
      const auto& num_dims = proto_deconv.num_dims();
      const auto& dim = proto_deconv.conv_dims_i();
      const auto& pad = proto_deconv.conv_pads_i();
      const auto& stride = proto_deconv.conv_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, bias, cudnn
                   );
      }
    }
  }

  // Transform layers
  if (proto_layer.has_reshape()) {
    const auto& proto_reshape = proto_layer.reshape();
    std::vector<int> dims = parse_list<int>(proto_reshape.dims());
    if (proto_layer.num_neurons_from_data_reader()) {
      dims.clear();
      if (proto_reshape.reshape_to_flattened_conv_format()) {
        dims.push_back(1);
      }
      dims.push_back(data_readers[execution_mode::training]->get_linearized_data_size());
    }
    return new reshape_layer<layout>(comm, dims.size(), dims.data());
  }
  if (proto_layer.has_sum()) {
    const auto& scaling_factors = parse_list<DataType>(proto_layer.sum().scaling_factors());
    return new sum_layer<layout>(comm, scaling_factors, cudnn);
  }
  if (proto_layer.has_split()) {
    return new split_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_concatenation()) {
    const auto& axis = proto_layer.concatenation().concatenation_axis();
    return new concatenation_layer<layout>(comm, axis, cudnn);
  }
  if (proto_layer.has_slice()) {
    const auto& proto_slice = proto_layer.slice();
    const auto& axis = proto_slice.slice_axis();
    const auto& slice_points = parse_list<int>(proto_slice.slice_points());
    return new slice_layer<layout>(comm, axis, slice_points, cudnn);
  }
  if (proto_layer.has_hadamard()) {
    return new hadamard_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_constant()) {
    const auto& proto_constant = proto_layer.constant();
    const auto& dims = parse_list<int>(proto_constant.num_neurons());
    const auto& value = proto_constant.value();
    return new constant_layer<layout>(comm, value, dims, cudnn);
  }
  if (proto_layer.has_noise()) {
    const auto& proto_noise = proto_layer.noise();
    const auto& dims = parse_list<int>(proto_noise.num_neurons());
    const auto& noise_factor = proto_noise.noise_factor();
    return new noise_layer<layout>(comm, dims, noise_factor, cudnn);
  }
  if (proto_layer.has_pooling()) {
    const auto& proto_pool = proto_layer.pooling();
    const auto& mode_str = proto_pool.pool_mode();
    pool_mode mode = pool_mode::invalid;
    if (mode_str == "max" )            { mode = pool_mode::max; }
    if (mode_str == "average" )        { mode = pool_mode::average; }
    if (mode_str == "average_no_pad" ) { mode = pool_mode::average_no_pad; }
    if (proto_pool.has_vectors()) {
      const auto& dims = parse_list<int>(proto_pool.pool_dims());
      const auto& pads = parse_list<int>(proto_pool.pool_pads());
      const auto& strides = parse_list<int>(proto_pool.pool_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new pooling_layer<data_layout::DATA_PARALLEL>(
                     comm, dims.size(), dims, pads, strides, mode, cudnn
                   );
      }
    } else {
      const auto& num_dims = proto_pool.num_dims();
      const auto& dim = proto_pool.pool_dims_i();
      const auto& pad = proto_pool.pool_pads_i();
      const auto& stride = proto_pool.pool_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new pooling_layer<data_layout::DATA_PARALLEL>(
                     comm, num_dims, dim, pad, stride, mode, cudnn
                   );
      }
    }
  }
  if (proto_layer.has_unpooling()) {
    if (layout == data_layout::DATA_PARALLEL) {
      return new unpooling_layer<data_layout::DATA_PARALLEL>(comm);
    }
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& proto_bn = proto_layer.batch_normalization();
    const auto& decay = proto_bn.decay();
    const auto& epsilon = proto_bn.epsilon();
    const auto& global_stats = proto_bn.global_stats();
    if (layout == data_layout::DATA_PARALLEL) {
      return new batch_normalization<data_layout::DATA_PARALLEL>(
                   comm, decay, epsilon, global_stats, cudnn
                 );
    }
  }
  if (proto_layer.has_dropout()) {
    const auto& keep_prob = proto_layer.dropout().keep_prob();
    return new dropout<layout>(comm, keep_prob);
  }
  if (proto_layer.has_local_response_normalization()) {
    const auto& proto_lrn = proto_layer.local_response_normalization();
    const auto& alpha = proto_lrn.lrn_alpha();
    const auto& beta = proto_lrn.lrn_beta();
    const auto& k = proto_lrn.lrn_k();
    const auto& window_width = proto_lrn.window_width();
    if (layout == data_layout::DATA_PARALLEL) {
      return new local_response_normalization_layer<data_layout::DATA_PARALLEL>(
                   comm, window_width, alpha, beta, k, cudnn
                 );
    }
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& proto_seludropout = proto_layer.selu_dropout();
    const auto& keep_prob = proto_seludropout.keep_prob();
    const auto& alpha = proto_seludropout.alpha();
    const auto& scale = proto_seludropout.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return new selu_dropout<layout>(comm, keep_prob, alpha, scale);
    } else {
      return new selu_dropout<layout>(comm, keep_prob);
    }
  }

  // Activation layers
  if (proto_layer.has_softmax()) {
    return new softmax_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_relu()) {
    return new relu_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_sigmoid()) {
    return new sigmoid_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_tanh()) {
    return new tanh_layer<layout>(comm);
  }
  if (proto_layer.has_atan()) {
    return new atan_layer<layout>(comm);
  }
  if (proto_layer.has_exponential()) {
    return new exponential_layer<layout>(comm);
  }
  if (proto_layer.has_identity()) {
    return new identity_layer<layout>(comm);
  }
  if (proto_layer.has_bent_identity()) {
    return new bent_identity_layer<layout>(comm);
  }
  if (proto_layer.has_softplus()) {
    return new softplus_layer<layout>(comm);
  }
  if (proto_layer.has_smooth_relu()) {
    return new smooth_relu_layer<layout>(comm);
  }
  if (proto_layer.has_leaky_relu()) {
    return new leaky_relu_layer<layout>(comm);
  }
  if (proto_layer.has_swish()) {
    return new swish_layer<layout>(comm);
  }
  if (proto_layer.has_elu()) {
    const auto& alpha = proto_layer.elu().alpha();
    return new elu_layer<layout>(comm, alpha);
  }
  if (proto_layer.has_selu()) {
    const auto& proto_selu = proto_layer.selu();
    const auto& alpha = proto_selu.alpha();
    const auto& scale = proto_selu.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return new selu_layer<layout>(comm, alpha, scale);
    } else {
      return new selu_layer<layout>(comm);
    }
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(comm, err.str());
  return nullptr;

}

// Template instantiation
template Layer* construct_layer<data_layout::DATA_PARALLEL>(
  lbann_comm* comm,
  std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  cudnn::cudnn_manager* cudnn,
  const lbann_data::Layer& proto_layer
);
template Layer* construct_layer<data_layout::MODEL_PARALLEL>(
  lbann_comm* comm,
  std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  cudnn::cudnn_manager* cudnn,
  const lbann_data::Layer& proto_layer
);

} // namespace proto
} // namespace lbann
