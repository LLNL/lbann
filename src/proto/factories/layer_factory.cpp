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
    const auto& params = proto_layer.input();
    const auto& io_buffer = params.io_buffer();
    if (io_buffer == "distributed") {
      return new input_layer<distributed_io_buffer, layout>(comm,
                                                            num_parallel_readers,
                                                            data_readers,
                                                            !params.data_set_per_model(),
                                                            params.for_regression());
    }
    if (io_buffer == "partitioned") {
      return new input_layer<partitioned_io_buffer, layout>(comm,
                                                            num_parallel_readers,
                                                            data_readers,
                                                            !params.data_set_per_model(),
                                                            params.for_regression());
    }
  }
  if (proto_layer.has_repeated_input()) {
    /// @todo Remove when possible
    const auto& params = proto_layer.repeated_input();
    return new repeated_input_layer(comm,
                                    num_parallel_readers,
                                    data_readers,
                                    params.num_steps(),
                                    !params.data_set_per_model(),
                                    params.for_regression());
  }

  // Target layers
  if (proto_layer.has_target()) {
    const auto& params = proto_layer.target();
    const auto& io_buffer = params.io_buffer();
    if (io_buffer == "distributed") {
      return new target_layer<distributed_io_buffer, layout>(comm,
                                                             nullptr,
                                                             num_parallel_readers,
                                                             data_readers,
                                                             params.shared_data_reader(),
                                                             params.for_regression());
    }
    if (io_buffer == "partitioned") {
      return new target_layer<partitioned_io_buffer, layout>(comm,
                                                             nullptr,
                                                             num_parallel_readers,
                                                             data_readers,
                                                             params.shared_data_reader(),
                                                             params.for_regression());
    }
  }
  if (proto_layer.has_reconstruction()) {
    return new reconstruction_layer<layout>(comm, nullptr);
  }

  // Fully connected layer
  if (proto_layer.has_fully_connected()) {
    const auto& params = proto_layer.fully_connected();
    int num_neurons = params.num_neurons();
    if (proto_layer.num_neurons_from_data_reader()) {
      num_neurons = data_readers[execution_mode::training]->get_linearized_data_size();
    }
    return new fully_connected_layer<layout>(comm,
                                             num_neurons,
                                             nullptr,
                                             params.has_bias(),
                                             cudnn);
  }

  // Convolution and deconvolution layer
  if (proto_layer.has_convolution()) {
    const auto& params = proto_layer.convolution();
    const auto& num_output_channels = params.num_output_channels();
    const auto& bias = params.has_bias();
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, bias, cudnn
                   );
      }
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, bias, cudnn
                   );
      }
    }
  }
  if (proto_layer.has_deconvolution()) {
    const auto& params = proto_layer.deconvolution();
    const auto& bias = params.has_bias();
    int num_output_channels = params.num_output_channels();
    if (proto_layer.num_neurons_from_data_reader()) {
      num_output_channels = data_readers[execution_mode::training]->get_linearized_data_size();
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, bias, cudnn
                   );
      }
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
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
    const auto& params = proto_layer.reshape();
    std::vector<int> dims = parse_list<int>(params.dims());
    if (proto_layer.num_neurons_from_data_reader()) {
      dims.clear();
      if (params.reshape_to_flattened_conv_format()) {
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
    const auto& params = proto_layer.slice();
    const auto& slice_points = parse_list<int>(params.slice_points());
    return new slice_layer<layout>(comm,
                                   params.slice_axis(),
                                   slice_points,
                                   cudnn);
  }
  if (proto_layer.has_hadamard()) {
    return new hadamard_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_constant()) {
    const auto& params = proto_layer.constant();
    const auto& dims = parse_list<int>(params.num_neurons());
    return new constant_layer<layout>(comm, params.value(), dims, cudnn);
  }
  if (proto_layer.has_gaussian()) {
    const auto& params = proto_layer.gaussian();
    const auto& dims = parse_list<int>(params.neuron_dims());
    return new gaussian_layer<layout>(comm,
                                      dims,
                                      params.mean(),
                                      params.stdev(),
                                      cudnn);
  }
  if (proto_layer.has_bernoulli()) {
    const auto& params = proto_layer.bernoulli();
    const auto& dims = parse_list<int>(params.neuron_dims());
    return new bernoulli_layer<layout>(comm,
                                       dims,
                                       params.prob(),
                                       cudnn);
  }
  if (proto_layer.has_uniform()) {
    const auto& params = proto_layer.uniform();
    const auto& dims = parse_list<int>(params.neuron_dims());
    return new uniform_layer<layout>(comm,
                                     dims,
                                     params.min(),
                                     params.max(),
                                     cudnn);
  }
  if (proto_layer.has_pooling()) {
    const auto& params = proto_layer.pooling();
    const auto& mode_str = params.pool_mode();
    pool_mode mode = pool_mode::invalid;
    if (mode_str == "max" )            { mode = pool_mode::max; }
    if (mode_str == "average" )        { mode = pool_mode::average; }
    if (mode_str == "average_no_pad" ) { mode = pool_mode::average_no_pad; }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.pool_dims());
      const auto& pads = parse_list<int>(params.pool_pads());
      const auto& strides = parse_list<int>(params.pool_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new pooling_layer<data_layout::DATA_PARALLEL>(
                     comm, dims.size(), dims, pads, strides, mode, cudnn
                   );
      }
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.pool_dims_i();
      const auto& pad = params.pool_pads_i();
      const auto& stride = params.pool_strides_i();
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
  if (proto_layer.has_reduction()) {
    const auto& params = proto_layer.reduction();
    const auto& mode_str = params.mode();
    reduction_mode mode = reduction_mode::INVALID;
    if (mode_str == "sum" || mode_str.empty()) { mode = reduction_mode::SUM; }
    if (mode_str == "average") { mode = reduction_mode::AVERAGE; }
    if (layout == data_layout::DATA_PARALLEL) {
      return new reduction_layer<data_layout::DATA_PARALLEL>(comm, mode, cudnn);
    }
  }
  if (proto_layer.has_evaluation()) {
    return new evaluation_layer<layout>(comm, cudnn);
  }
  if (proto_layer.has_crop()) {
    const auto& params = proto_layer.crop();
    const auto& dims = parse_list<int>(params.dims());
    if (layout == data_layout::DATA_PARALLEL) {
      return new crop_layer<data_layout::DATA_PARALLEL>(comm, dims, cudnn);
    }
  }
  if (proto_layer.has_categorical_random()) {
    if (layout == data_layout::DATA_PARALLEL) {
      return new categorical_random_layer<data_layout::DATA_PARALLEL>(comm, cudnn);
    }
  }
  if (proto_layer.has_discrete_random()) {
    const auto& params = proto_layer.discrete_random();
    const auto& values = parse_list<DataType>(params.values());
    const auto& dims = parse_list<int>(params.dims());
    if (layout == data_layout::DATA_PARALLEL) {
      return new discrete_random_layer<data_layout::DATA_PARALLEL>(comm, values, dims, cudnn);
    }
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& params = proto_layer.batch_normalization();
    if (layout == data_layout::DATA_PARALLEL) {
      return new batch_normalization<data_layout::DATA_PARALLEL>(comm,
                                                                 params.decay(),
                                                                 params.epsilon(),
                                                                 params.global_stats(),
                                                                 cudnn);
    }
  }
  if (proto_layer.has_dropout()) {
    const auto& params = proto_layer.dropout();
    return new dropout<layout>(comm, params.keep_prob(), cudnn);
  }
  if (proto_layer.has_local_response_normalization()) {
    const auto& params = proto_layer.local_response_normalization();
    if (layout == data_layout::DATA_PARALLEL) {
      return new local_response_normalization_layer<data_layout::DATA_PARALLEL>(comm,
                                                                                params.window_width(),
                                                                                params.lrn_alpha(),
                                                                                params.lrn_beta(),
                                                                                params.lrn_k(),
                                                                                cudnn);
    }
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& params = proto_layer.selu_dropout();
    const auto& keep_prob = params.keep_prob();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
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
    return new tanh_layer<layout>(comm, cudnn);
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
    const auto& params = proto_layer.elu();
    return new elu_layer<layout>(comm, params.alpha());
  }
  if (proto_layer.has_selu()) {
    const auto& params = proto_layer.selu();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return new selu_layer<layout>(comm, alpha, scale);
    } else {
      return new selu_layer<layout>(comm);
    }
  }
  if (proto_layer.has_power()) {
    const auto& params = proto_layer.power();
    return new power_layer<layout>(comm, params.exponent());
  }
  if (proto_layer.has_log()) {
    const auto& params = proto_layer.log();
    const auto& base = params.base();
    if (base != 0.0) {
      return new log_layer<layout>(comm, base);
    } else {
      return new log_layer<layout>(comm);
    }
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(err.str());
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
