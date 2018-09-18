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
#include "lbann/models/sequential.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/models/greedy_layerwise_autoencoder.hpp"
#include "lbann/models/siamese.hpp"

/// Activation Layers
#include "lbann/layers/activations/atan.hpp"
#include "lbann/layers/activations/bent_identity.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/exponential.hpp"
#include "lbann/layers/activations/identity.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/selu.hpp"
#include "lbann/layers/activations/sigmoid.hpp"
#include "lbann/layers/activations/smooth_relu.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/activations/logsoftmax.hpp"
#include "lbann/layers/activations/softplus.hpp"
#include "lbann/layers/activations/swish.hpp"
#include "lbann/layers/activations/tanh.hpp"
#include "lbann/layers/activations/power.hpp"
#include "lbann/layers/activations/sigmoid_bce_with_logits.hpp"
#include "lbann/layers/activations/abs.hpp"
#include "lbann/layers/activations/l2_loss.hpp"
#include "lbann/layers/activations/log.hpp"

/// Learning Layers
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"

/// Loss Layers
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"

/// Transform Layers
#include "lbann/layers/transform/reshape.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/layers/transform/unpooling.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/sum.hpp"
#include "lbann/layers/transform/weighted_sum.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/concatenation.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/safe_inv.hpp"
#include "lbann/layers/transform/hadamard.hpp"
#include "lbann/layers/transform/zero.hpp"
#include "lbann/layers/transform/reduction.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/layers/transform/bernoulli.hpp"
#include "lbann/layers/transform/uniform.hpp"
#include "lbann/layers/transform/crop.hpp"
#include "lbann/layers/transform/categorical_random.hpp"
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/layers/transform/stop_gradient.hpp"
#include "lbann/layers/transform/max.hpp"
#include "lbann/layers/transform/min.hpp"
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/layers/transform/sort.hpp"
#include "lbann/layers/transform/weights.hpp"

/// Regularization layers.
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/layers/regularizers/selu_dropout.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"

/// I/O Layers
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/target/target_layer.hpp"

/// Reconstruction Layer
#include "lbann/layers/io/target/reconstruction.hpp"

/// Data Readers
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/data_readers/data_reader_imagenet_patches.hpp"
#include "lbann/data_readers/data_reader_cifar10.hpp"
#include "lbann/data_readers/data_reader_mnist.hpp"
#include "lbann/data_readers/data_reader_multi_images.hpp"
#include "lbann/data_readers/data_reader_mnist_siamese.hpp"
#include "lbann/data_readers/data_reader_triplet.hpp"
#include "lbann/data_readers/data_reader_synthetic.hpp"
#include "lbann/data_readers/data_reader_jag.hpp"
#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include "lbann/data_readers/data_reader_nci.hpp"
#include "lbann/data_readers/data_reader_numpy.hpp"
#include "lbann/data_readers/data_reader_csv.hpp"
#include "lbann/data_readers/data_reader_merge_samples.hpp"
#include "lbann/data_readers/data_reader_merge_features.hpp"
#include "lbann/data_readers/data_reader_ascii.hpp"
#include "lbann/data_readers/data_reader_pilot2_molecular.hpp"
#include "lbann/data_readers/data_reader_mesh.hpp"

/// Data Store
#include "lbann/data_store/generic_data_store.hpp"
#include "lbann/data_store/data_store_imagenet.hpp"
//#include "lbann/data_store/data_store_merge_samples.hpp"

/// Callbacks
#include "lbann/callbacks/callback_check_init.hpp"
#include "lbann/callbacks/callback_checknan.hpp"
#include "lbann/callbacks/callback_checksmall.hpp"
#include "lbann/callbacks/callback_check_dataset.hpp"
#include "lbann/callbacks/callback_print.hpp"
#include "lbann/callbacks/callback_timer.hpp"
#include "lbann/callbacks/callback_io.hpp"
#include "lbann/callbacks/callback_summary.hpp"
#include "lbann/callbacks/callback_learning_rate.hpp"
#include "lbann/callbacks/callback_debug.hpp"
#include "lbann/callbacks/callback_debug_io.hpp"
#include "lbann/callbacks/callback_imcomm.hpp"
#include "lbann/callbacks/callback_dump_weights.hpp"
#include "lbann/callbacks/callback_dump_activations.hpp"
#include "lbann/callbacks/callback_dump_error_signals.hpp"
#include "lbann/callbacks/callback_dump_gradients.hpp"
#include "lbann/callbacks/callback_dump_minibatch_sample_indices.hpp"
#include "lbann/callbacks/callback_early_stopping.hpp"
#include "lbann/callbacks/callback_ltfb.hpp"
#include "lbann/callbacks/callback_save_images.hpp"
#include "lbann/callbacks/profiler.hpp"
#include "lbann/callbacks/callback_check_reconstruction_error.hpp"
#include "lbann/callbacks/callback_hang.hpp"
#include "lbann/callbacks/callback_variable_minibatch.hpp"
#include "lbann/callbacks/callback_gradient_check.hpp"
#include "lbann/callbacks/callback_timeline.hpp"
#include "lbann/callbacks/callback_checkpoint.hpp"
#include "lbann/callbacks/callback_save_model.hpp"
#include "lbann/callbacks/callback_replace_weights.hpp"
#include "lbann/callbacks/callback_gpu_memory_usage.hpp"
#include "lbann/callbacks/callback_sync_layers.hpp"
#include "lbann/callbacks/callback_sync_selected.hpp"

/// Weights and weight initializers
#include "lbann/weights/weights.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

/// Optimizers
#include "lbann/optimizers/adagrad.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/optimizers/hypergradient_adam.hpp"
#include "lbann/optimizers/rmsprop.hpp"
#include "lbann/optimizers/sgd.hpp"

/// Objective functions
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/objective_functions/loss_functions/cross_entropy.hpp"
#include "lbann/objective_functions/loss_functions/mean_squared_error.hpp"
#include "lbann/objective_functions/loss_functions/binary_cross_entropy.hpp"
#include "lbann/objective_functions/loss_functions/cross_entropy_with_uncertainty.hpp"
#include "lbann/objective_functions/loss_functions/geom_negloglike.hpp"
#include "lbann/objective_functions/loss_functions/mean_absolute_deviation.hpp"
#include "lbann/objective_functions/loss_functions/mean_absolute_error.hpp"
#include "lbann/objective_functions/loss_functions/poisson_negloglike.hpp"
#include "lbann/objective_functions/loss_functions/polya_negloglike.hpp"
#include "lbann/objective_functions/weight_regularization/l1.hpp"
#include "lbann/objective_functions/weight_regularization/l2.hpp"
#include "lbann/objective_functions/weight_regularization/group_lasso.hpp"
#include "lbann/objective_functions/layer_term.hpp"

/// Metrics
#include "lbann/metrics/layer_metric.hpp"
#include "lbann/metrics/categorical_accuracy.hpp"
#include "lbann/metrics/top_k_categorical_accuracy.hpp"
#include "lbann/metrics/mean_squared_error.hpp"
#include "lbann/metrics/mean_absolute_deviation.hpp"
#include "lbann/metrics/pearson_correlation.hpp"
#include "lbann/metrics/r2.hpp"
#include "lbann/metrics/boolean_accuracy.hpp"
#include "lbann/metrics/boolean_false_positives.hpp"
#include "lbann/metrics/boolean_false_negatives.hpp"

/// Utilities, exceptions, etc.
#include "lbann/utils/exception.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/glob.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/compiler_control.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/peek_map.hpp"

#endif // LBANN_HPP_INCLUDED
