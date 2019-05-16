# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  callback_check_dataset.cpp
  callback_check_gradients.cpp
  callback_check_init.cpp
  callback_check_metric.cpp
  callback_checknan.cpp
  callback_checkpoint.cpp
  callback_checksmall.cpp
  callback_confusion_matrix.cpp
  callback_debug.cpp
  callback_debug_io.cpp
  callback_dump_outputs.cpp
  callback_dump_error_signals.cpp
  callback_dump_gradients.cpp
  callback_dump_minibatch_sample_indices.cpp
  callback_dump_weights.cpp
  callback_early_stopping.cpp
  callback_imcomm.cpp
  callback_io.cpp
  callback_learning_rate.cpp
  callback_ltfb.cpp
  callback_perturb_adam.cpp
  callback_print.cpp
  callback_save_images.cpp
  callback_save_model.cpp
  callback_summary.cpp
  callback_sync_layers.cpp
  callback_sync_selected.cpp
  callback_timeline.cpp
  callback_timer.cpp
  callback_variable_minibatch.cpp
  profiler.cpp
  callback_replace_weights.cpp
  callback_gpu_memory_usage.cpp
  callback_perturb_dropout.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
