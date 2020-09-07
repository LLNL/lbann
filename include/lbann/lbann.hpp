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

#ifndef LBANN_LBANN_HPP_INCLUDED
#define LBANN_LBANN_HPP_INCLUDED

/// Trainers
#include "lbann/trainers/trainer.hpp"

/// Training Algorithms
#include "lbann/training_algorithms/training_algorithm.hpp"

/// Models
#include "lbann/models/directed_acyclic_graph.hpp"

/// Activation layers
#include "lbann/layers/activations/activations.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/identity.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/softmax.hpp"

/// Image layers
#include "lbann/layers/image/bilinear_resize.hpp"

/// Learning layers
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"
#include "lbann/layers/learning/embedding.hpp"
#include "lbann/layers/learning/channelwise_scale_bias.hpp"
#include "lbann/layers/learning/entrywise_scale_bias.hpp"

/// Loss layers
#include "lbann/layers/loss/categorical_accuracy.hpp"
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/layers/loss/entrywise.hpp"
#include "lbann/layers/loss/l1_norm.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"
#include "lbann/layers/loss/mean_absolute_error.hpp"
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"

/// Math layers
#include "lbann/layers/math/unary.hpp"
#include "lbann/layers/math/binary.hpp"
#include "lbann/layers/math/clamp.hpp"

/// Transform layers
#include "lbann/layers/transform/reshape.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/layers/transform/unpooling.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/sum.hpp"
#include "lbann/layers/transform/weighted_sum.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/concatenate.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/hadamard.hpp"
#include "lbann/layers/transform/reduction.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/layers/transform/bernoulli.hpp"
#include "lbann/layers/transform/uniform.hpp"
#include "lbann/layers/transform/crop.hpp"
#include "lbann/layers/transform/categorical_random.hpp"
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/layers/transform/stop_gradient.hpp"
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/layers/transform/sort.hpp"
#include "lbann/layers/transform/weights.hpp"
#include "lbann/layers/transform/tessellate.hpp"

/// Regularization layers
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/layers/regularizers/selu_dropout.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/layers/regularizers/instance_norm.hpp"

/// Input layer
#include "lbann/layers/io/input/input_layer.hpp"

/// Miscellaneous layers
#include "lbann/layers/misc/covariance.hpp"
#include "lbann/layers/misc/variance.hpp"
#include "lbann/layers/misc/channelwise_mean.hpp"
#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/layers/misc/mini_batch_index.hpp"
#include "lbann/layers/misc/mini_batch_size.hpp"
#include "lbann/layers/misc/argmax.hpp"
#include "lbann/layers/misc/argmin.hpp"
#include "lbann/layers/misc/one_hot.hpp"

/// Data readers
#include "lbann/data_readers/data_reader_npz_ras_lipid.hpp"
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/data_readers/data_reader_cifar10.hpp"
#include "lbann/data_readers/data_reader_mnist.hpp"
#include "lbann/data_readers/data_reader_synthetic.hpp"
#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/data_readers/data_reader_nci.hpp"
#include "lbann/data_readers/data_reader_numpy.hpp"
#include "lbann/data_readers/data_reader_numpy_npz.hpp"
#include "lbann/data_readers/data_reader_numpy_npz_conduit.hpp"
#include "lbann/data_readers/data_reader_csv.hpp"
#include "lbann/data_readers/data_reader_merge_samples.hpp"
#include "lbann/data_readers/data_reader_merge_features.hpp"
#include "lbann/data_readers/data_reader_pilot2_molecular.hpp"
#include "lbann/data_readers/data_reader_mesh.hpp"
#include "lbann/data_readers/data_reader_python.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/data_readers/data_reader_hdf5.hpp"
#endif // LBANN_HAS_DISTCONV
#include "lbann/data_readers/data_reader_smiles.hpp"

/// Data stores
#include "lbann/data_store/data_store_conduit.hpp"

/// Callbacks
#include "lbann/callbacks/check_dataset.hpp"
#include "lbann/callbacks/check_gradients.hpp"
#include "lbann/callbacks/check_init.hpp"
#include "lbann/callbacks/check_metric.hpp"
#include "lbann/callbacks/check_nan.hpp"
#include "lbann/callbacks/check_small.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/confusion_matrix.hpp"
#include "lbann/callbacks/debug.hpp"
#include "lbann/callbacks/debug_io.hpp"
#include "lbann/callbacks/dump_error_signals.hpp"
#include "lbann/callbacks/dump_gradients.hpp"
#include "lbann/callbacks/dump_minibatch_sample_indices.hpp"
#include "lbann/callbacks/dump_outputs.hpp"
#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/early_stopping.hpp"
#include "lbann/callbacks/gpu_memory_usage.hpp"
#include "lbann/callbacks/hang.hpp"
#include "lbann/callbacks/imcomm.hpp"
#include "lbann/callbacks/learning_rate.hpp"
#include "lbann/callbacks/ltfb.hpp"
#include "lbann/callbacks/mixup.hpp"
#include "lbann/callbacks/monitor_io.hpp"
#include "lbann/callbacks/perturb_adam.hpp"
#include "lbann/callbacks/perturb_dropout.hpp"
#include "lbann/callbacks/print_model_description.hpp"
#include "lbann/callbacks/print_statistics.hpp"
#include "lbann/callbacks/profiler.hpp"
#include "lbann/callbacks/replace_weights.hpp"
#include "lbann/callbacks/save_images.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/callbacks/load_model.hpp"
#include "lbann/callbacks/save_topk_models.hpp"
#include "lbann/callbacks/summary.hpp"
#include "lbann/callbacks/sync_layers.hpp"
#include "lbann/callbacks/timeline.hpp"
#include "lbann/callbacks/timer.hpp"
#include "lbann/callbacks/variable_minibatch.hpp"

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
#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/objective_functions/weight_regularization/l2.hpp"

/// Metrics
#include "lbann/metrics/layer_metric.hpp"

/// Utilities, exceptions, library interface, etc.
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/glob.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/compiler_control.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/peek_map.hpp"
#include "lbann/utils/stack_trace.hpp"
#include "lbann/utils/stack_profiler.hpp"

#endif // LBANN_LBANN_HPP_INCLUDED
