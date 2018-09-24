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
#include "lbann/utils/peek_map.hpp"

namespace lbann {
namespace proto {

template <data_layout layout, El::Device Dev>
Layer* construct_layer(lbann_comm* comm,
                       const std::map<execution_mode, generic_data_reader*>& data_readers,
                       int num_parallel_readers,
                       const lbann_data::Layer& proto_layer) {
  std::stringstream err;

  // Convenience macro to construct layers with no parameters
#define CONSTRUCT_LAYER(name)                           \
  do {                                                  \
    if (proto_layer.has_##name()) {                     \
      return new name##_layer<layout, Dev>(comm);       \
    }                                                   \
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
    if (io_buffer == "distributed") {
      return new input_layer<distributed_io_buffer, layout, Dev>(comm,
                                                                 num_parallel_readers,
                                                                 data_readers,
                                                                 !params.data_set_per_model(),
                                                                 target_mode);
    }
    if (io_buffer == "partitioned") {
      return new input_layer<partitioned_io_buffer, layout, Dev>(comm,
                                                                 num_parallel_readers,
                                                                 data_readers,
                                                                 !params.data_set_per_model(),
                                                                 target_mode);
    }
  }

  // Target layers
  if (proto_layer.has_target()) {
    return new target_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_reconstruction()) {
    return new reconstruction_layer<layout, Dev>(comm);
  }

  // Fully connected layer
  if (proto_layer.has_fully_connected()) {
    const auto& params = proto_layer.fully_connected();
    int num_neurons = 0;
    if (params.get_input_dimension_from_reader() 
        || params.get_image_dimension_from_reader()
        || params.get_scalar_dimension_from_reader())
       {
    #if defined(LBANN_HAS_CONDUIT)
       const auto dr1  = lbann::peek_map(data_readers, execution_mode::training);
       lbann::data_reader_jag_conduit_hdf5 *dr = dynamic_cast<lbann::data_reader_jag_conduit_hdf5*>(dr1);
       size_t input_dim = dr->get_linearized_input_size();
       size_t scalar_dim = dr->get_linearized_scalar_size();
       size_t image_dim = dr->get_linearized_image_size();
       size_t num_images = dr->get_num_img_srcs();

       if (params.get_input_dimension_from_reader()) {
         num_neurons += input_dim;
       }
       if (params.get_image_dimension_from_reader()) {
         num_neurons += (num_images * image_dim);
       }
       if (params.get_scalar_dimension_from_reader()) {
         num_neurons += scalar_dim;
       }
    #else
      err << "get_*_dimension_from_reader() not supported";
      LBANN_ERROR(err.str());
      return nullptr;
    #endif // defined(LBANN_HAS_CONDUIT)
    } else {
      num_neurons = params.num_neurons();
      if (proto_layer.num_neurons_from_data_reader()) {
        const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
        if (!dr) {
          LBANN_ERROR("training data reader does not exist!");
        }
        num_neurons = dr->get_linearized_data_size();
      }
    }
    return new fully_connected_layer<layout, Dev>(comm,
                                                  num_neurons,
                                                  params.transpose(),
                                                  nullptr,
                                                  params.has_bias());
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
        return new convolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, bias
                   );
      }
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, bias
                   );
      }
    }
  }
  if (proto_layer.has_deconvolution()) {
    const auto& params = proto_layer.deconvolution();
    const auto& bias = params.has_bias();
    int num_output_channels = params.num_output_channels();
    if (proto_layer.num_neurons_from_data_reader()) {
      const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
      if (!dr) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      num_output_channels = dr->get_linearized_data_size();
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, bias
                   );
      }
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, bias
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
      const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
      if (!dr) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      dims.push_back(dr->get_linearized_data_size());
    }
    return new reshape_layer<layout, Dev>(comm, dims);
  }
  if (proto_layer.has_sum()) {
    return new sum_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_weighted_sum()) {
    const auto& params = proto_layer.weighted_sum();
    const auto& scaling_factors = parse_list<DataType>(params.scaling_factors());
    return new weighted_sum_layer<layout, Dev>(comm, scaling_factors);
  }
  if (proto_layer.has_split()) {
    return new split_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_concatenation()) {
    const auto& axis = proto_layer.concatenation().concatenation_axis();
    return new concatenation_layer<layout, Dev>(comm, axis);
  }
  if (proto_layer.has_slice()) {
    const auto& params = proto_layer.slice();
    if (params.get_slice_points_from_reader() != "") {
    #if defined(LBANN_HAS_CONDUIT)
      std::stringstream ss;
      ss << params.get_slice_points_from_reader();
      std::string s;
      std::vector<El::Int> slice_points;
      size_t total = 0;
      slice_points.push_back(total);
      const auto dr1  = lbann::peek_map(data_readers, execution_mode::training);
      lbann::data_reader_jag_conduit_hdf5 *dr = dynamic_cast<lbann::data_reader_jag_conduit_hdf5*>(dr1);
      while (ss >> s) {
        if (s != "") {  //probably not needed
          if (s == "scalars") {
            total += dr->get_linearized_scalar_size();
            slice_points.push_back(total);
          } else if (s == "images") {
            total += dr->get_num_img_srcs() * dr->get_linearized_image_size();
            slice_points.push_back(total);
          } else if (s == "inputs") {
            total += dr->get_linearized_input_size();
            slice_points.push_back(total);
          } else {
            err << __FILE__ << " " << __LINE__ << " :: "
                << "unknown string in slice layer for get_slice_points_from_reader(): " << s << "; should be scalars, images, or inputs\n";
            throw lbann_exception(err.str());
          }
        }
      }
      return new slice_layer<layout, Dev>(comm,
                                          params.slice_axis(),
                                          slice_points);
    #else
      err << "get_slice_points_from_reader() not supported";
      LBANN_ERROR(err.str());
      return nullptr;
    #endif // defined(LBANN_HAS_CONDUIT)
    } else {
      const auto& slice_points = parse_list<El::Int>(params.slice_points());
      return new slice_layer<layout, Dev>(comm,
                                          params.slice_axis(),
                                          slice_points);
    }
  }
  if (proto_layer.has_hadamard()) {
    return new hadamard_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_constant()) {
    const auto& params = proto_layer.constant();
    const auto& dims = parse_list<int>(params.num_neurons());
    return new constant_layer<layout, Dev>(comm, params.value(), dims);
  }
  if (proto_layer.has_gaussian()) {
    const auto& params = proto_layer.gaussian();
    const auto& dims = parse_list<int>(params.neuron_dims());
    if (params.mean() == 0 && params.stdev() == 0) {
      return new gaussian_layer<layout, Dev>(comm, dims);
    } else {
      return new gaussian_layer<layout, Dev>(comm,
                                             dims,
                                             params.mean(),
                                             params.stdev());
    }
  }
  if (proto_layer.has_bernoulli()) {
    const auto& params = proto_layer.bernoulli();
    const auto& dims = parse_list<int>(params.neuron_dims());
    return new bernoulli_layer<layout, Dev>(comm,
                                            dims,
                                            params.prob());
  }
  if (proto_layer.has_uniform()) {
    const auto& params = proto_layer.uniform();
    const auto& dims = parse_list<int>(params.neuron_dims());
    if (params.min() == 0 && params.max() == 0) {
      return new uniform_layer<layout, Dev>(comm, dims);
    } else {
      return new uniform_layer<layout, Dev>(comm, dims, params.min(), params.max());
    }
  }
  if (proto_layer.has_zero()) {
    const auto& params = proto_layer.zero();
    return new zero_layer<layout>(comm, params.first_half(), params.second_half());
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
        return new pooling_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, dims.size(), dims, pads, strides, mode
                   );
      }
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.pool_dims_i();
      const auto& pad = params.pool_pads_i();
      const auto& stride = params.pool_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new pooling_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, num_dims, dim, pad, stride, mode
                   );
      }
    }
  }
  if (proto_layer.has_unpooling()) {
    if (layout == data_layout::DATA_PARALLEL && Dev == El::Device::CPU) {
      return new unpooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(comm);
    }
  }
  if (proto_layer.has_reduction()) {
    const auto& params = proto_layer.reduction();
    const auto& mode_str = params.mode();
    reduction_mode mode = reduction_mode::INVALID;
    if (mode_str == "sum" || mode_str.empty()) { mode = reduction_mode::SUM; }
    if (mode_str == "average") { mode = reduction_mode::AVERAGE; }
    if (layout == data_layout::DATA_PARALLEL) {
      return new reduction_layer<data_layout::DATA_PARALLEL, Dev>(comm, mode);
    }
  }
  if (proto_layer.has_evaluation()) {
    return new evaluation_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_crop()) {
    const auto& params = proto_layer.crop();
    const auto& dims = parse_list<int>(params.dims());
    if (layout == data_layout::DATA_PARALLEL) {
      return new crop_layer<data_layout::DATA_PARALLEL, Dev>(comm, dims);
    }
  }
  if (proto_layer.has_categorical_random()) {
    if (layout == data_layout::DATA_PARALLEL
        && Dev == El::Device::CPU) {
      return new categorical_random_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(comm);
    }
  }
  if (proto_layer.has_discrete_random()) {
    const auto& params = proto_layer.discrete_random();
    const auto& values = parse_list<DataType>(params.values());
    const auto& dims = parse_list<int>(params.dims());
    if (layout == data_layout::DATA_PARALLEL
        && Dev == El::Device::CPU) {
      return new discrete_random_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                   comm, values, dims);
    }
  }
  if (proto_layer.has_dummy()) {
    return new dummy_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_stop_gradient()) {
    return new stop_gradient_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_in_top_k()) {
    const auto& params = proto_layer.in_top_k();
    return new in_top_k_layer<layout, Dev>(comm, params.k());
  }
  if (proto_layer.has_sort()) {
    const auto& params = proto_layer.sort();
    if (layout == data_layout::DATA_PARALLEL) {
      return new sort_layer<data_layout::DATA_PARALLEL, Dev>(comm, params.descending());
    }
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& params = proto_layer.batch_normalization();
    if (layout == data_layout::DATA_PARALLEL) {
      return new batch_normalization_layer<data_layout::DATA_PARALLEL, Dev>(comm,
                                                                            params.decay(),
                                                                            params.epsilon(),
                                                                            params.global_stats());
    } else {
      LBANN_ERROR("batch normalization is only supported in a data-parallel layout");
    }
  }
  if (proto_layer.has_dropout()) {
    const auto& params = proto_layer.dropout();
    return new dropout<layout, Dev>(comm, params.keep_prob());
  }
  if (proto_layer.has_local_response_normalization()) {
 const auto& params = proto_layer.local_response_normalization();
    if (layout == data_layout::DATA_PARALLEL) {
      return new local_response_normalization_layer<data_layout::DATA_PARALLEL, Dev>(comm,
                                                                                     params.window_width(),
                                                                                     params.lrn_alpha(),
                                                                                     params.lrn_beta(),
                                                                                     params.lrn_k());
    }
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& params = proto_layer.selu_dropout();
    const auto& keep_prob = params.keep_prob();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return new selu_dropout<layout, Dev>(comm, keep_prob, alpha, scale);
    } else {
      return new selu_dropout<layout, Dev>(comm, keep_prob);
    }
  }

  // Math layers
  if (proto_layer.has_not_()) { return new not_layer<layout, Dev>(comm); }
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
  CONSTRUCT_LAYER(max);
  CONSTRUCT_LAYER(min);
  CONSTRUCT_LAYER(equal);
  CONSTRUCT_LAYER(not_equal);
  CONSTRUCT_LAYER(less);
  CONSTRUCT_LAYER(less_equal);
  CONSTRUCT_LAYER(greater);
  CONSTRUCT_LAYER(greater_equal);
  if (proto_layer.has_and_()) { return new and_layer<layout, Dev>(comm); }
  if (proto_layer.has_or_())  { return new or_layer<layout, Dev>(comm); }
  if (proto_layer.has_xor_()) { return new xor_layer<layout, Dev>(comm); }
  
  // Activation layers
  if (proto_layer.has_softmax()) {
    return new softmax_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_logsoftmax()) {
    return new logsoftmax_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_relu()) {
    return new relu_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_sigmoid()) {
    return new sigmoid_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_identity()) {
    return new identity_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_bent_identity()) {
    return new bent_identity_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_softplus()) {
    return new softplus_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_smooth_relu()) {
    return new smooth_relu_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_leaky_relu()) {
    return new leaky_relu_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_swish()) {
    return new swish_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_elu()) {
    const auto& params = proto_layer.elu();
    return new elu_layer<layout, Dev>(comm, params.alpha());
  }
  if (proto_layer.has_selu()) {
    const auto& params = proto_layer.selu();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return new selu_layer<layout, Dev>(comm, alpha, scale);
    } else {
      return new selu_layer<layout, Dev>(comm);
    }
  }
  if (proto_layer.has_l2_loss()) {
    return new l2_loss_layer<layout, Dev>(comm);
  }

  // Loss layers
  if (proto_layer.has_cross_entropy()) {
    return new cross_entropy_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_mean_squared_error()) {
    return new mean_squared_error_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_top_k_categorical_accuracy()) {
    const auto& params = proto_layer.top_k_categorical_accuracy();
    return new top_k_categorical_accuracy_layer<layout, Dev>(comm, params.k());
  }

  if (proto_layer.has_bce_with_logits()) {
    const auto& params = proto_layer.bce_with_logits();
    return new sigmoid_bce_with_logits_layer<layout, Dev>(comm, params.true_label());
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(err.str());
  return nullptr;

}

// Template instantiation
template Layer* construct_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template Layer* construct_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#ifdef LBANN_HAS_GPU
template Layer* construct_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template Layer* construct_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#endif // LBANN_HAS_GPU

} // namespace proto
} // namespace lbann
