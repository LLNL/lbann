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

/// Layers
#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/layers/lbann_layer_softmax.hpp"
#include "lbann/layers/lbann_layer_convolutional.hpp"
#include "lbann/layers/lbann_layer_pooling.hpp"

/// I/O Layers
#include "lbann/layers/lbann_input_layer_distributed_minibatch.hpp"
#include "lbann/layers/lbann_target_layer_distributed_minibatch.hpp"
#include "lbann/layers/lbann_input_layer_distributed_minibatch_parallel_io.hpp"
#include "lbann/layers/lbann_target_layer_distributed_minibatch_parallel_io.hpp"
//#include "lbann/layers/lbann_target_layer_unsupervised.hpp"

/// Data Readers
#include "lbann/data_readers/lbann_data_reader_imagenet.hpp"

/// Callbacks
#include "lbann/callbacks/lbann_callback_print.hpp"
#include "lbann/callbacks/lbann_callback_io.hpp"
#include "lbann/callbacks/lbann_callback_summary.hpp"
#include "lbann/callbacks/lbann_callback_timer.hpp"
#include "lbann/callbacks/lbann_callback_learning_rate.hpp"
#include "lbann/callbacks/lbann_callback_debug.hpp"
#include "lbann/callbacks/lbann_callback_imcomm.hpp"
#include "lbann/callbacks/lbann_callback_dump_weights.hpp"

/// Objective functions (cost functions)
#include "lbann/objective_functions/lbann_objective_fn.hpp"
#include "lbann/objective_functions/lbann_objective_fn_categorical_cross_entropy.hpp"
#include "lbann/objective_functions/lbann_objective_fn_mean_squared_error.hpp"


/// Regularizers
#include "lbann/regularization/lbann_dropout.hpp"

/// Utilities, exceptions, etc.
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/utils/lbann_summary.hpp"
#include "lbann/lbann_params.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/io/lbann_persist.hpp"

#endif // LBANN_HPP_INCLUDED
