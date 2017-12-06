# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  callback_check_dataset.cpp
  callback_check_init.cpp
  callback_check_reconstruction_error.cpp
  callback_checknan.cpp
  callback_checksmall.cpp
  callback_debug.cpp
  callback_debug_io.cpp
  callback_dump_activations.cpp
  callback_dump_gradients.cpp
  callback_dump_minibatch_sample_indices.cpp
  callback_dump_weights.cpp
  callback_early_stopping.cpp
  callback_gradient_check.cpp
  callback_imcomm.cpp
  callback_io.cpp
  callback_learning_rate.cpp
  callback_ltfb.cpp
  callback_print.cpp
  callback_save_images.cpp
  callback_summary.cpp
  callback_timeline.cpp
  callback_timer.cpp
  callback_variable_minibatch.cpp
  profiler.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
