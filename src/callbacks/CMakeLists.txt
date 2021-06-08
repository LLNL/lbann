# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
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
  gpu_memory_usage.cpp
  hang.cpp
  imcomm.cpp
  learning_rate.cpp
  load_model.cpp
  ltfb.cpp
  mixup.cpp
  monitor_io.cpp
  perturb_adam.cpp
  perturb_dropout.cpp
  perturb_learning_rate.cpp
  print_model_description.cpp
  print_statistics.cpp
  profiler.cpp
  replace_weights.cpp
  save_images.cpp
  save_model.cpp
  save_topk_models.cpp
  set_weights_value.cpp
  setup_communitygan_data_reader.cpp
  summary.cpp
  summarize_images.cpp
  sync_layers.cpp
  timeline.cpp
  timer.cpp
  variable_minibatch.cpp
)

# Add the subdirectories
add_subdirectory(kfac)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
