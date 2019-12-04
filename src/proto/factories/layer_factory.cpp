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

#include "lbann/proto/factories.hpp"

#include "lbann/layers/layer.hpp"
#include "lbann/layers/activations/activations.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/identity.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/image/bilinear_resize.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/io_layer.hpp"
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/channelwise_scale_bias.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"
#include "lbann/layers/learning/embedding.hpp"
#include "lbann/layers/learning/entrywise_scale_bias.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/loss/categorical_accuracy.hpp"
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/layers/loss/entrywise.hpp"
#include "lbann/layers/loss/l1_norm.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"
#include "lbann/layers/loss/mean_absolute_error.hpp"
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include "lbann/layers/math/binary.hpp"
#include "lbann/layers/math/clamp.hpp"
#include "lbann/layers/math/matmul.hpp"
#include "lbann/layers/math/unary.hpp"
#include "lbann/layers/misc/channelwise_mean.hpp"
#include "lbann/layers/misc/covariance.hpp"
#include "lbann/layers/misc/mini_batch_index.hpp"
#include "lbann/layers/misc/mini_batch_size.hpp"
#include "lbann/layers/misc/variance.hpp"
#include "lbann/layers/misc/argmax.hpp"
#include "lbann/layers/misc/argmin.hpp"
#include "lbann/layers/misc/one_hot.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/layers/regularizers/selu_dropout.hpp"
#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/layers/transform/bernoulli.hpp"
#include "lbann/layers/transform/categorical_random.hpp"
#include "lbann/layers/transform/concatenation.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/transform/crop.hpp"
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/layers/transform/hadamard.hpp"
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/layers/transform/reduction.hpp"
#include "lbann/layers/transform/reshape.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/sort.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/stop_gradient.hpp"
#include "lbann/layers/transform/sum.hpp"
#include "lbann/layers/transform/tessellate.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/layers/transform/uniform.hpp"
#include "lbann/layers/transform/unpooling.hpp"
#include "lbann/layers/transform/weighted_sum.hpp"
#include "lbann/layers/transform/weights.hpp"

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/peek_map.hpp"

#include <layers.pb.h>

namespace lbann {
namespace proto {

std::vector<El::Int> get_slice_points_from_reader(const generic_data_reader* dr,
                                                  const std::string& var_category,
                                                  bool& is_supported);

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer) {
  std::stringstream err;

  // Convenience macro to construct layers with no parameters
#define CONSTRUCT_LAYER(name)                                           \
  do {                                                                         \
    if (proto_layer.has_##name()) {                                            \
      return lbann::make_unique<name##_layer<TensorDataType, Layout, Device>>(comm); \
    }                                                                          \
  } while (false)

  // Input layers
  if (proto_layer.has_input()) {
    const auto& params = proto_layer.input();
    const auto& io_buffer = params.io_buffer();
    const auto& mode_str = params.target_mode();
    data_reader_target_mode target_mode = data_reader_target_mode::CLASSIFICATION;
    if (mode_str.empty() || mode_str == "classification") { target_mode = data_reader_target_mode::CLASSIFICATION; }
    if (mode_str == "regression")                         { target_mode = data_reader_target_mode::REGRESSION; }
    if (mode_str == "reconstruction")                     { target_mode = data_reader_target_mode::RECONSTRUCTION; }
    if (mode_str == "na" || mode_str == "NA" || mode_str == "N/A") { target_mode = data_reader_target_mode::NA; }
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("input layer is only supported with "
                  "a data-parallel layout");
    }
    if (io_buffer == "partitioned" || io_buffer.empty()) {
      /// @todo Question for Tim Moon and Tom Benson, I had to change this line from Layout to
      /// data_layout::DATA_PARALLEL to make it compile with clang on OS X, but it seems like
      /// this is not related to this PR.
      return lbann::make_unique<input_layer<TensorDataType,partitioned_io_buffer<TensorDataType>,data_layout::DATA_PARALLEL,Device>>(
               comm,
               num_parallel_readers,
               data_readers,
               !params.data_set_per_model(),
               target_mode);
    } else {
      LBANN_ERROR("invalid IO buffer type (" + io_buffer + ")");
    }
  }

  // Fully connected layer
  if (proto_layer.has_fully_connected()) {
    const auto& params = proto_layer.fully_connected();
    return lbann::make_unique<fully_connected_layer<TensorDataType, Layout, Device>>(
             comm,
             params.num_neurons(),
             params.transpose(),
             nullptr,
             params.has_bias());
  }

  // Convolution and deconvolution layer
  if (proto_layer.has_convolution()) {
    const auto& params = proto_layer.convolution();
    const auto& num_output_channels = params.num_output_channels();
    const auto& bias = params.has_bias();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("convolution layer is only supported with "
                  "a data-parallel layout");
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
      return lbann::make_unique<convolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, dims.size(), num_output_channels,
               dims, pads, strides, dilations, num_groups, bias);
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
      return lbann::make_unique<convolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, num_dims, num_output_channels,
               dim, pad, stride, dilation, num_groups, bias);
    }
  }
  if (proto_layer.has_deconvolution()) {
    const auto& params = proto_layer.deconvolution();
    const auto& bias = params.has_bias();
    int num_output_channels = params.num_output_channels();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
      if (!dr) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      num_output_channels = dr->get_linearized_data_size();
    }
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("deconvolution layer is only supported with "
                  "a data-parallel layout");
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
      return lbann::make_unique<deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, dims.size(), num_output_channels,
               dims, pads, strides, dilations, num_groups, bias);
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
      return lbann::make_unique<deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, num_dims, num_output_channels,
               dim, pad, stride, dilation, num_groups, bias);
    }
  }

  // Learning layers
  if (proto_layer.has_embedding()) {
    const auto& params = proto_layer.embedding();
    const size_t num_embeddings = params.num_embeddings();
    const size_t embedding_dim = params.embedding_dim();
    const El::Int padding_idx = (params.has_padding_idx() ?
                                 params.padding_idx().value() : -1);
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<embedding_layer<TensorDataType, data_layout::DATA_PARALLEL,Device>>(
        comm, num_embeddings, embedding_dim, padding_idx);
    } else {
      LBANN_ERROR("embedding layer is only supported with "
                  "data-parallel data layout");
    }
  }
  if (proto_layer.has_channelwise_scale_bias()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<channelwise_scale_bias_layer<TensorDataType, data_layout::DATA_PARALLEL,Device>>(comm);
    } else {
      LBANN_ERROR("channel-wise scale/bias layer is only supported "
                  "with data-parallel data layout");
    }
  }
  if (proto_layer.has_entrywise_scale_bias()) {
    return lbann::make_unique<entrywise_scale_bias_layer<TensorDataType, Layout,Device>>(comm);
  }

  // Transform layers
  if (proto_layer.has_reshape()) {
    const auto& params = proto_layer.reshape();
    std::vector<int> dims = parse_list<int>(params.dims());
    if (params.num_dims() != 0) {
      LBANN_WARNING("found unused and deprecated prototext field (Reshape.num_dims)");
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      dims.clear();
      const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
      if (!dr) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      dims.push_back(dr->get_linearized_data_size());
    }
    return lbann::make_unique<reshape_layer<TensorDataType, Layout, Device>>(comm, dims);
  }
  if (proto_layer.has_sum()) {
    return lbann::make_unique<sum_layer<TensorDataType, Layout, Device>>(comm);
  }
  if (proto_layer.has_weighted_sum()) {
    const auto& params = proto_layer.weighted_sum();
    const auto& scaling_factors = parse_list<DataType>(params.scaling_factors());
    return lbann::make_unique<weighted_sum_layer<TensorDataType, Layout, Device>>(comm, scaling_factors);
  }
  if (proto_layer.has_split()) {
    return lbann::make_unique<split_layer<TensorDataType, Layout, Device>>(comm);
  }
  if (proto_layer.has_concatenation()) {
    const auto& axis = proto_layer.concatenation().axis();
    return lbann::make_unique<concatenation_layer<TensorDataType, Layout, Device>>(comm, axis);
  }
  if (proto_layer.has_slice()) {
    const auto& params = proto_layer.slice();
    std::vector<El::Int> slice_points;
    bool is_supported = false;
    std::string slice_point_method_name;

    if (params.get_slice_points_from_reader() != "") {
      slice_point_method_name = "'get_slice_points_from_reader'";
      const auto dr_generic  = lbann::peek_map(data_readers, execution_mode::training);
      const std::string& var = params.get_slice_points_from_reader();
      slice_points = get_slice_points_from_reader(dr_generic, var, is_supported);
    } else {
      slice_point_method_name = "'slice_points'";
      slice_points = parse_list<El::Int>(params.slice_points());
      is_supported = true;
    }
    if (slice_points.size() < 2u) {
      if (is_supported) {
        err << "Failed to get slice points via " << slice_point_method_name << '.';
      } else {
        err << slice_point_method_name << " is not supported by the reader.";
      }
      LBANN_ERROR(err.str());
      return nullptr;
    }
    return lbann::make_unique<slice_layer<TensorDataType, Layout, Device>>(
             comm, params.axis(), slice_points);
  }
  if (proto_layer.has_hadamard()) {
    return lbann::make_unique<hadamard_layer<TensorDataType, Layout, Device>>(comm);
  }
  if (proto_layer.has_constant()) {
    const auto& params = proto_layer.constant();
    const auto& dims = parse_list<int>(params.num_neurons());
    return lbann::make_unique<constant_layer<TensorDataType, Layout, Device>>(comm, params.value(), dims);
  }
  if (proto_layer.has_gaussian()) {
    const auto& params = proto_layer.gaussian();
    const auto& dims = parse_list<int>(params.neuron_dims());
    if (params.mean() == 0 && params.stdev() == 0) {
      return lbann::make_unique<gaussian_layer<TensorDataType, Layout, Device>>(comm, dims);
    } else {
      return lbann::make_unique<gaussian_layer<TensorDataType, Layout, Device>>(comm,
                                             dims,
                                             params.mean(),
                                             params.stdev());
    }
  }
  if (proto_layer.has_bernoulli()) {
    const auto& params = proto_layer.bernoulli();
    const auto& dims = parse_list<int>(params.neuron_dims());
    return lbann::make_unique<bernoulli_layer<TensorDataType, Layout, Device>>(
             comm, dims, params.prob());
  }
  if (proto_layer.has_uniform()) {
    const auto& params = proto_layer.uniform();
    const auto& dims = parse_list<int>(params.neuron_dims());
    if (params.min() == 0 && params.max() == 0) {
      return lbann::make_unique<uniform_layer<TensorDataType, Layout, Device>>(comm, dims);
    } else {
      return lbann::make_unique<uniform_layer<TensorDataType, Layout, Device>>(
               comm, dims, params.min(), params.max());
    }
  }
  if (proto_layer.has_pooling()) {
    const auto& params = proto_layer.pooling();
    const auto& mode_str = params.pool_mode();
    pool_mode mode = pool_mode::invalid;
    if (mode_str == "max" )            { mode = pool_mode::max; }
    if (mode_str == "average" )        { mode = pool_mode::average; }
    if (mode_str == "average_no_pad" ) { mode = pool_mode::average_no_pad; }
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("pooling layer is only supported with "
                  "a data-parallel layout");
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.pool_dims());
      const auto& pads = parse_list<int>(params.pool_pads());
      const auto& strides = parse_list<int>(params.pool_strides());
      return lbann::make_unique<pooling_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, dims.size(), dims, pads, strides, mode);
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.pool_dims_i();
      const auto& pad = params.pool_pads_i();
      const auto& stride = params.pool_strides_i();
      return lbann::make_unique<pooling_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, num_dims, dim, pad, stride, mode);
    }
  }
  if (proto_layer.has_unpooling()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return lbann::make_unique<unpooling_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("unpooling layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_reduction()) {
    const auto& params = proto_layer.reduction();
    const auto& mode_str = params.mode();
    reduction_mode mode = reduction_mode::INVALID;
    if (mode_str == "sum" || mode_str.empty()) { mode = reduction_mode::SUM; }
    if (mode_str == "average") { mode = reduction_mode::AVERAGE; }
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<reduction_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, mode);
    } else {
      LBANN_ERROR("reduction layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_evaluation()) {
    return lbann::make_unique<evaluation_layer<TensorDataType, Layout, Device>>(comm);
  }
  if (proto_layer.has_crop()) {
    const auto& params = proto_layer.crop();
    const auto& dims = parse_list<int>(params.dims());
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<crop_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, dims);
    } else {
      LBANN_ERROR("crop layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_categorical_random()) {
    if (Layout == data_layout::DATA_PARALLEL
        && Device == El::Device::CPU) {
      return lbann::make_unique<categorical_random_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("categorical random layer is only supported on CPU");
    }
  }
  if (proto_layer.has_discrete_random()) {
    const auto& params = proto_layer.discrete_random();
    const auto& values = parse_list<DataType>(params.values());
    const auto& dims = parse_list<int>(params.dims());
    if (Layout == data_layout::DATA_PARALLEL
        && Device == El::Device::CPU) {
      return lbann::make_unique<discrete_random_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(
               comm, values, dims);
    } else {
      LBANN_ERROR("discrete random layer is only supported on CPU");
    }
  }
  if (proto_layer.has_dummy()) {
    return lbann::make_unique<dummy_layer<TensorDataType, Layout, Device>>(comm);
  }
  if (proto_layer.has_stop_gradient()) {
    return lbann::make_unique<stop_gradient_layer<TensorDataType, Layout, Device>>(comm);
  }
  if (proto_layer.has_in_top_k()) {
    const auto& params = proto_layer.in_top_k();
    return lbann::make_unique<in_top_k_layer<TensorDataType, Layout, Device>>(comm, params.k());
  }
  if (proto_layer.has_sort()) {
    const auto& params = proto_layer.sort();
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<sort_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, params.descending());
    } else {
      LBANN_ERROR("sort layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_weights_layer()) {
    const auto& params = proto_layer.weights_layer();
    const auto& dims = parse_list<El::Int>(params.dims());
    return lbann::make_unique<weights_layer<TensorDataType, Layout, Device>>(comm, dims);
  }
  if (proto_layer.has_tessellate()) {
    const auto& params = proto_layer.tessellate();
    const auto& dims = parse_list<int>(params.dims());
    return lbann::make_unique<tessellate_layer<TensorDataType, Layout, Device>>(comm, dims);
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& params = proto_layer.batch_normalization();
    if (Layout == data_layout::DATA_PARALLEL) {
      int statistics_group_size = params.statistics_group_size();
      if (statistics_group_size < 0) {
        statistics_group_size = 0;  // Global statistics.
      } else if (statistics_group_size == 0) {
        statistics_group_size = 1;  // Default to local.
      }
      const auto& aggr_str = params.stats_aggregation();
      if (!aggr_str.empty()) {
        LBANN_WARNING("stats_aggregation field for BatchNormalization is deprecated");
        if (aggr_str == "local") {
          statistics_group_size = 1;
        } else if (aggr_str == "node_local") {
          statistics_group_size = comm->get_procs_per_node();
        } else if (aggr_str == "global") {
          statistics_group_size = 0;
        } else {
          err << "Invalid batch normalization stats aggregation " << aggr_str;
          LBANN_ERROR(err.str());
          return nullptr;
        }
      }
      // Set defaults if not given.
      auto decay = params.decay();
      if (decay == 0.0) {
        decay = 0.9;
      }
      auto epsilon = params.epsilon();
      if (epsilon == 0.0) {
        epsilon = 1e-5;
      }
      return lbann::make_unique<batch_normalization_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
        comm,
        decay,
        epsilon,
        statistics_group_size);
    } else {
      LBANN_ERROR("batch normalization layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_dropout()) {
    const auto& params = proto_layer.dropout();
    return lbann::make_unique<dropout<TensorDataType, Layout, Device>>(comm, params.keep_prob());
  }
  if (proto_layer.has_local_response_normalization()) {
 const auto& params = proto_layer.local_response_normalization();
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<local_response_normalization_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
             comm,
             params.window_width(),
             params.lrn_alpha(),
             params.lrn_beta(),
             params.lrn_k());
    } else {
      LBANN_ERROR("local response normalization layer is only supported "
                  "with a data-parallel layout");
    }
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& params = proto_layer.selu_dropout();
    const auto& keep_prob = params.keep_prob();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return lbann::make_unique<selu_dropout<TensorDataType, Layout, Device>>(comm, keep_prob, alpha, scale);
    } else {
      return lbann::make_unique<selu_dropout<TensorDataType, Layout, Device>>(comm, keep_prob);
    }
  }
  if (proto_layer.has_entrywise_batch_normalization()) {
    const auto& params = proto_layer.entrywise_batch_normalization();
    return lbann::make_unique<entrywise_batch_normalization_layer<TensorDataType, Layout, Device>>(comm, params.decay(), params.epsilon());
  }
  if (proto_layer.has_layer_norm()) {
    const auto& params = proto_layer.layer_norm();
    const double epsilon = (params.has_epsilon()
                            ? params.epsilon().value()
                            : 1e-5);
    return lbann::make_unique<layer_norm_layer<TensorDataType, Layout, Device>>(comm, epsilon);
  }

  // Math layers
  CONSTRUCT_LAYER(logical_not);
  CONSTRUCT_LAYER(abs);
  CONSTRUCT_LAYER(negative);
  CONSTRUCT_LAYER(sign);
  CONSTRUCT_LAYER(round);
  CONSTRUCT_LAYER(ceil);
  CONSTRUCT_LAYER(floor);
  CONSTRUCT_LAYER(reciprocal);
  CONSTRUCT_LAYER(square);
  CONSTRUCT_LAYER(sqrt);
  CONSTRUCT_LAYER(rsqrt);
  CONSTRUCT_LAYER(safe_reciprocal);
  CONSTRUCT_LAYER(exp);
  CONSTRUCT_LAYER(expm1);
  CONSTRUCT_LAYER(log);
  CONSTRUCT_LAYER(log1p);
  CONSTRUCT_LAYER(cos);
  CONSTRUCT_LAYER(sin);
  CONSTRUCT_LAYER(tan);
  CONSTRUCT_LAYER(acos);
  CONSTRUCT_LAYER(asin);
  CONSTRUCT_LAYER(atan);
  CONSTRUCT_LAYER(cosh);
  CONSTRUCT_LAYER(sinh);
  CONSTRUCT_LAYER(tanh);
  CONSTRUCT_LAYER(acosh);
  CONSTRUCT_LAYER(asinh);
  CONSTRUCT_LAYER(atanh);
  CONSTRUCT_LAYER(add);
  CONSTRUCT_LAYER(subtract);
  CONSTRUCT_LAYER(multiply);
  CONSTRUCT_LAYER(divide);
  CONSTRUCT_LAYER(mod);
  CONSTRUCT_LAYER(pow);
  CONSTRUCT_LAYER(safe_divide);
  CONSTRUCT_LAYER(squared_difference);
  CONSTRUCT_LAYER(max);
  CONSTRUCT_LAYER(min);
  CONSTRUCT_LAYER(equal);
  CONSTRUCT_LAYER(not_equal);
  CONSTRUCT_LAYER(less);
  CONSTRUCT_LAYER(less_equal);
  CONSTRUCT_LAYER(greater);
  CONSTRUCT_LAYER(greater_equal);
  CONSTRUCT_LAYER(logical_and);
  CONSTRUCT_LAYER(logical_or);
  CONSTRUCT_LAYER(logical_xor);
  if (proto_layer.has_clamp()) {
    const auto& params = proto_layer.clamp();
    return lbann::make_unique<clamp_layer<TensorDataType, Layout, Device>>(comm, params.min(), params.max());
  }
  if (proto_layer.has_matmul()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,Device>>(comm);
    } else {
      LBANN_ERROR("matrix multiply layer is only supported with "
                  "a data-parallel layout");
    }
  }

  // Activation layers
  if (proto_layer.has_elu()) {
    const auto& params = proto_layer.elu();
    const auto& alpha = params.alpha();
    if (alpha != 0) {
      return lbann::make_unique<elu_layer<TensorDataType, Layout, Device>>(comm, alpha);
    } else {
      return lbann::make_unique<elu_layer<TensorDataType, Layout, Device>>(comm);
    }
  }
  CONSTRUCT_LAYER(identity);
  if (proto_layer.has_leaky_relu()) {
    const auto& params = proto_layer.leaky_relu();
    const auto& negative_slope = params.negative_slope();
    if (negative_slope != 0) {
      return lbann::make_unique<leaky_relu_layer<TensorDataType, Layout, Device>>(comm, negative_slope);
    } else {
      return lbann::make_unique<leaky_relu_layer<TensorDataType, Layout, Device>>(comm);
    }
  }
  CONSTRUCT_LAYER(log_sigmoid);
  CONSTRUCT_LAYER(log_softmax);
  CONSTRUCT_LAYER(relu);
  CONSTRUCT_LAYER(selu);
  CONSTRUCT_LAYER(sigmoid);
  CONSTRUCT_LAYER(softmax);
  CONSTRUCT_LAYER(softplus);
  CONSTRUCT_LAYER(softsign);

  // Loss layers
  CONSTRUCT_LAYER(categorical_accuracy);
  CONSTRUCT_LAYER(cross_entropy);
  CONSTRUCT_LAYER(mean_squared_error);
  CONSTRUCT_LAYER(mean_absolute_error);
  if (proto_layer.has_top_k_categorical_accuracy()) {
    const auto& params = proto_layer.top_k_categorical_accuracy();
    return lbann::make_unique<top_k_categorical_accuracy_layer<TensorDataType, Layout, Device>>(comm, params.k());
  }
  CONSTRUCT_LAYER(l2_norm2);
  CONSTRUCT_LAYER(l1_norm);
  CONSTRUCT_LAYER(binary_cross_entropy);
  CONSTRUCT_LAYER(sigmoid_binary_cross_entropy);
  CONSTRUCT_LAYER(boolean_accuracy);
  CONSTRUCT_LAYER(boolean_false_negative);
  CONSTRUCT_LAYER(boolean_false_positive);

  // Image layers
  if (proto_layer.has_bilinear_resize()) {
    const auto& params = proto_layer.bilinear_resize();
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<bilinear_resize_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, params.height(), params.width());
    } else {
      LBANN_ERROR("bilinear resize layer is only supported with "
                  "a data-parallel layout");
    }
  }

  // Miscellaneous layers
  if (proto_layer.has_covariance()) {
    const auto& params = proto_layer.covariance();
    return lbann::make_unique<covariance_layer<TensorDataType, Layout, Device>>(comm, params.biased());
  }
  if (proto_layer.has_variance()) {
    const auto& params = proto_layer.variance();
    return lbann::make_unique<variance_layer<TensorDataType, Layout, Device>>(comm, params.biased());
  }
  if (proto_layer.has_channelwise_mean()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<channelwise_mean_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm);
    } else {
      LBANN_ERROR("channel-wise mean layer is only supported with "
                  "a data-parallel layout");
    }
  }
  CONSTRUCT_LAYER(mini_batch_index);
  CONSTRUCT_LAYER(mini_batch_size);
  if (proto_layer.has_argmax()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return lbann::make_unique<argmax_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("argmax layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_argmin()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return lbann::make_unique<argmin_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("argmin layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_one_hot()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      const auto& params = proto_layer.one_hot();
      return lbann::make_unique<one_hot_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, params.size());
    } else {
      LBANN_ERROR("one-hot layer is only supported with "
                  "a data-parallel layout");
    }
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(err.str());
  return nullptr;

}

// Template instantiation
template std::unique_ptr<Layer> construct_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template std::unique_ptr<Layer> construct_layer<DataType, data_layout::MODEL_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#ifdef LBANN_HAS_GPU
template std::unique_ptr<Layer> construct_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template std::unique_ptr<Layer> construct_layer<DataType, data_layout::MODEL_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#endif // LBANN_HAS_GPU

/// Obtain the slice points from the data reader
std::vector<El::Int> get_slice_points_from_reader(const generic_data_reader* dr_generic,
                                                  const std::string& var_category,
                                                  bool& is_supported) {
  std::vector<El::Int> slice_points;
  is_supported = false;
  // TODO: remove the dynamic cast when this feature gets merged into the base class
  const auto dr = dynamic_cast<const data_reader_jag_conduit*>(dr_generic);

  if (dr != nullptr) {
    is_supported = true;
    if (var_category == "independent") {
      slice_points = dr->get_slice_points_independent();
    } else if (var_category == "dependent") {
      slice_points = dr->get_slice_points_independent();
    } else {
      LBANN_ERROR("Unknown variable category \"" + var_category \
                  + "\". Must be either \"independent\" or \"dependent\".");
    }
  }
  return slice_points;
}

} // namespace proto
} // namespace lbann
