# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  callback.hpp
  callback_check_dataset.hpp
  callback_check_gradients.hpp
  callback_check_init.hpp
  callback_check_metric.hpp
  callback_checknan.hpp
  callback_checksmall.hpp
  callback_confusion_matrix.hpp
  callback_debug.hpp
  callback_debug_io.hpp
  callback_dump_outputs.hpp
  callback_dump_error_signals.hpp
  callback_dump_gradients.hpp
  callback_dump_minibatch_sample_indices.hpp
  callback_dump_weights.hpp
  callback_early_stopping.hpp
  callback_hang.hpp
  callback_imcomm.hpp
  callback_io.hpp
  callback_learning_rate.hpp
  callback_ltfb.hpp
  callback_mixup.hpp
  callback_perturb_adam.hpp
  callback_print.hpp
  callback_save_images.hpp
  callback_save_model.hpp
  callback_summary.hpp
  callback_timer.hpp
  callback_variable_minibatch.hpp
  profiler.hpp
  callback_gpu_memory_usage.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
