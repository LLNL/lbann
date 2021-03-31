# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  callback.hpp
  check_dataset.hpp
  check_gradients.hpp
  check_init.hpp
  check_metric.hpp
  check_nan.hpp
  check_small.hpp
  checkpoint.hpp
  confusion_matrix.hpp
  debug.hpp
  debug_io.hpp
  dump_error_signals.hpp
  dump_gradients.hpp
  dump_minibatch_sample_indices.hpp
  dump_model_graph.hpp
  dump_outputs.hpp
  dump_weights.hpp
  early_stopping.hpp
  gpu_memory_usage.hpp
  hang.hpp
  imcomm.hpp
  learning_rate.hpp
  ltfb.hpp
  mixup.hpp
  monitor_io.hpp
  perturb_adam.hpp
  perturb_dropout.hpp
  perturb_learning_rate.hpp
  print_model_description.hpp
  print_statistics.hpp
  profiler.hpp
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

# Add the subdirectories
add_subdirectory(kfac)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
