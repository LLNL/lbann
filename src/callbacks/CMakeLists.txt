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
# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  alternate_updates.cpp
  callback.cpp
  check_dataset.cpp
  check_gradients.cpp
  check_init.cpp
  check_metric.cpp
  check_nan.cpp
  check_small.cpp
  checkpoint.cpp
  confusion_matrix.cpp
  compute_model_size.cpp
  debug.cpp
  debug_io.cpp
  dump_error_signals.cpp
  dump_gradients.cpp
  dump_minibatch_sample_indices.cpp
  dump_model_graph.cpp
  dump_outputs.cpp
  dump_weights.cpp
  early_stopping.cpp
  evaluate_progress.cpp
  gpu_memory_usage.cpp
  gradient_clipping.cpp
  hang.cpp
  learning_rate.cpp
  load_model.cpp
  memory_profiler.cpp
  mixup.cpp
  monitor_io.cpp
  perturb_adam.cpp
  perturb_dropout.cpp
  perturb_learning_rate.cpp
  perturb_weights.cpp
  print_model_description.cpp
  print_statistics.cpp
  profiler.cpp
  progress_bar.cpp
  replace_weights.cpp
  save_images.cpp
  save_model.cpp
  save_topk_models.cpp
  set_weights_value.cpp
  summary.cpp
  summarize_images.cpp
  sync_layers.cpp
  timeline.cpp
  timer.cpp
  variable_minibatch.cpp
)

if(LBANN_HAS_ONNX)
  list(APPEND THIS_DIR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/export_onnx.cpp)
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
