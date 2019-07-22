# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  check_dataset.cpp
  check_gradients.cpp
  check_init.cpp
  check_metric.cpp
  check_nan.cpp
  check_small.cpp
  checkpoint.cpp
  confusion_matrix.cpp
  debug.cpp
  debug_io.cpp
  dump_error_signals.cpp
  dump_gradients.cpp
  dump_minibatch_sample_indices.cpp
  dump_outputs.cpp
  dump_weights.cpp
  early_stopping.cpp
  gpu_memory_usage.cpp
  hang.cpp
  imcomm.cpp
  io.cpp
  learning_rate.cpp
  ltfb.cpp
  mixup.cpp
  perturb_adam.cpp
  perturb_dropout.cpp
  print.cpp
  profiler.cpp
  replace_weights.cpp
  save_images.cpp
  save_model.cpp
  save_topk_models.cpp
  summary.cpp
  sync_layers.cpp
  sync_selected.cpp
  timeline.cpp
  timer.cpp
  variable_minibatch.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
