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
//
// lbann.hpp - LBANN top level header
////////////////////////////////////////////////////////////////////////////////

/**
 * Top level LBANN header
 *  - includes everything needed for the models in the model zoo
 */
#ifndef LBANN_HPP_INCLUDED
#define LBANN_HPP_INCLUDED

/// Models
#include "lbann/models/lbann_model_dnn.hpp"
#include "lbann/models/lbann_model_stacked_autoencoder.hpp"
#include "lbann/models/lbann_model_greedy_layerwise_autoencoder.hpp"

/// Activation Layers
#include "lbann/layers/activations/activations.hpp"
#include "lbann/layers/activations/create_activation.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/softplus.hpp"
#include "lbann/layers/activations/id.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/sigmoid.hpp"
#include "lbann/layers/activations/smooth_relu.hpp"
#include "lbann/layers/activations/tanh.hpp"

/// Layers
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/transform/pooling.hpp"

/// Regularization layers.
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"

/// I/O Layers
#include "lbann/layers/io/input/lbann_input_layer_distributed_minibatch.hpp"
#include "lbann/layers/io/target/lbann_target_layer_distributed_minibatch.hpp"
#include "lbann/layers/io/input/lbann_input_layer_distributed_minibatch_parallel_io.hpp"
#include "lbann/layers/io/target/lbann_target_layer_distributed_minibatch_parallel_io.hpp"
#include "lbann/layers/io/input/lbann_input_layer_partitioned_minibatch_parallel_io.hpp"
#include "lbann/layers/io/target/lbann_target_layer_partitioned_minibatch_parallel_io.hpp"
//#include "lbann/layers/lbann_target_layer_unsupervised.hpp"

/// Data Readers
#include "lbann/data_readers/lbann_data_reader_imagenet.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet_single.hpp"
#include "lbann/data_readers/lbann_data_reader_cifar10.hpp"
#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet_single_cv.hpp"
#include "lbann/data_readers/lbann_data_reader_synthetic.hpp"
#include "lbann/data_readers/lbann_data_reader_nci.hpp"
#include "lbann/data_readers/lbann_data_reader_nci_regression.hpp"
#include "lbann/data_readers/lbann_data_reader_cnpy.hpp"

/// Callbacks
#include "lbann/callbacks/lbann_callback_print.hpp"
#include "lbann/callbacks/lbann_callback_io.hpp"
#include "lbann/callbacks/lbann_callback_summary.hpp"
#include "lbann/callbacks/lbann_callback_timer.hpp"
#include "lbann/callbacks/lbann_callback_learning_rate.hpp"
#include "lbann/callbacks/lbann_callback_debug.hpp"
#include "lbann/callbacks/lbann_callback_imcomm.hpp"
#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/callbacks/lbann_callback_dump_minibatch_sample_indices.hpp"
#include "lbann/callbacks/lbann_callback_early_stopping.hpp"

/// Objective functions (cost functions)
#include "lbann/objective_functions/lbann_objective_fn.hpp"
#include "lbann/objective_functions/lbann_objective_fn_categorical_cross_entropy.hpp"
#include "lbann/objective_functions/lbann_objective_fn_mean_squared_error.hpp"

/// Metrics
#include "lbann/metrics/lbann_metric.hpp"
#include "lbann/metrics/lbann_metric_categorical_accuracy.hpp"
#include "lbann/metrics/lbann_metric_mean_squared_error.hpp"

/// Utilities, exceptions, etc.
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/utils/lbann_summary.hpp"
#include "lbann/lbann_params.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/io/lbann_persist.hpp"

#endif // LBANN_HPP_INCLUDED
