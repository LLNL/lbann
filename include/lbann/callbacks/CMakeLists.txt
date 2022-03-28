################################################################################
## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################
# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  alternate_updates.hpp
  callback.hpp
  check_dataset.hpp
  check_gradients.hpp
  check_init.hpp
  check_metric.hpp
  check_nan.hpp
  check_small.hpp
  checkpoint.hpp
  confusion_matrix.hpp
  compute_model_size.hpp
  debug.hpp
  debug_io.hpp
  dump_error_signals.hpp
  dump_gradients.hpp
  dump_minibatch_sample_indices.hpp
  dump_model_graph.hpp
  dump_outputs.hpp
  dump_weights.hpp
  early_stopping.hpp
  evaluate_progress.hpp
  gpu_memory_usage.hpp
  gradient_clipping.hpp
  hang.hpp
  learning_rate.hpp
  memory_profiler.hpp
  mixup.hpp
  monitor_io.hpp
  perturb_adam.hpp
  perturb_dropout.hpp
  perturb_learning_rate.hpp
  perturb_weights.hpp
  print_model_description.hpp
  print_statistics.hpp
  profiler.hpp
  progress_bar.hpp
  replace_weights.hpp
  save_images.hpp
  save_model.hpp
  save_topk_models.hpp
  set_weights_value.hpp
  summary.hpp
  sync_layers.hpp
  timeline.hpp
  timer.hpp
  variable_minibatch.hpp
  )

if(LBANN_HAS_ONNX)
  list(APPEND THIS_DIR_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/export_onnx.hpp)
endif ()
# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
