////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include "lbann/utils/options.hpp"
#include "lbann/utils/argument_parser.hpp"

namespace lbann {

void construct_std_options() {
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(DISABLE_BACKGROUND_IO_ACTIVITY,
                      {"--disable_background_io_activity"},
                      "prevent the input layers from fetching data in the background");
  arg_parser.add_flag(DISABLE_CUDA,
                      {"--disable_cuda"},
                      "has no effect unless LBANN was compiled with LBANN_HAS_CUDNN");
  arg_parser.add_flag(HELP,
                      {"--help", "-h"},
                      "Prints the help message");
  arg_parser.add_flag(LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE,
                      {"--load_model_weights_dir_is_complete"},
                      "Use load_model_weights_dir as given, ignoring checkpoint hierarchy");
  arg_parser.add_flag(LTFB_VERBOSE,
                      {"--ltfb_verbose"},
                      "Increases number of per-trainer messages that are reported");
  arg_parser.add_flag(PRELOAD_DATA_STORE,
                      {"--preload_data_store"},
                      "Preloads the data store in-memory structure druing data reader load time");
  arg_parser.add_flag(PRINT_AFFINITY,
                      {"--print_affinity"},
                      "display information on how OpenMP threads are provisioned");
  arg_parser.add_flag(SERIALIZE_IO,
                      {"--serialize_io"},
                      "force data readers to use a single threaded for I/O");
  arg_parser.add_flag(ST_FULL_TRACE,
                      {"--st_full_trace"},
                      "TODO");
  arg_parser.add_flag(ST_ON,
                      {"--st_on"},
                      "TODO");
	arg_parser.add_flag(USE_CUBLAS_TENSOR_OPS,
											{"--use-cublas-tensor-ops"},
											utils::ENV("LBANN_USE_CUBLAS_TENSOR_OPS"),
											"Set the default cuBLAS math mode to use "
											"Tensor Core operations when available.");
	arg_parser.add_flag(USE_CUDNN_TENSOR_OPS,
											{"--use-cudnn-tensor-ops"},
											utils::ENV("LBANN_USE_CUDNN_TENSOR_OPS"),
											"Set the default cuDNN math mode to use "
											"Tensor Core operations when available.");
  arg_parser.add_flag(USE_DATA_STORE,
                      {"--use_data_store"},
                      "Enables the data store in-memory structure");
  arg_parser.add_flag(USE_LTFB,
                      {"--ltfb"},
                      "TODO");
  arg_parser.add_flag(VERBOSE,
                      {"--verbose", "--verbose_print"},
                      "Turns on verbose mode");
  arg_parser.add_flag(WRITE_SAMPLE_LIST,
                      {"--write_sample_list"},
                      "Writes out the sample list that was loaded into the current directory");

  // Input options
  arg_parser.add_flag(ALLOW_GLOBAL_STATISTICS,
                      {"--ltfb_allow_global_statistics"},
                      utils::ENV("LBANN_LTFB_ALLOW_GLOBAL_STATISTICS"),
                      "Allow the print_statistics callback to report "
                      "global (inter-trainer) summary statistics.");
  arg_parser.add_option(HYDROGEN_BLOCK_SIZE,
                        {"--hydrogen_block_size"},
                        "Block size for Hydrogen",
                        0);
  arg_parser.add_option(LOAD_MODEL_WEIGHTS_DIR,
                        {"--load_model_weights_dir"},
                        "Load model wieghts found in the given directory.\n"
                        "If the directory doesn't exist, doesn't contain valid weights,\n"
                        "or doesn't contain a checkpoint,\n"
                        "an error will be thrown.\n",
                        "");
  arg_parser.add_option(MAX_RNG_SEEDS_DISPLAY,
                        {"--rng_seeds_per_trainer_to_display"},
                        utils::ENV("LBANN_RNG_SEEDS_PER_TRAINER_TO_DISPLAY"),
                        "Limit how many random seeds LBANN should display "
                        "from each trainer",
                        2);
  arg_parser.add_option(METADATA,
                        {"--metadata"},
                        "TODO",
                        "");
  arg_parser.add_option(MINI_BATCH_SIZE,
                        {"--mini_batch_size"},
                        "Size of mini batches",
                        -1);
  arg_parser.add_option(MODEL,
                        {"--model"},
                        "TODO",
                        "");
  arg_parser.add_option(NUM_EPOCHS,
                        {"--num_epochs"},
                        "Number of epochs to train model",
                        -1);
  arg_parser.add_option(NUM_IO_THREADS,
                        {"--num_io_threads"},
                        utils::ENV("LBANN_NUM_IO_THREADS"),
                        "Number of threads available to both I/O and "
                        "initial data transformations for each rank.",
                        64);
  arg_parser.add_option(NUM_PARALLEL_READERS,
                        {"--num_parallel_readers"},
                        "The number of parallel data readers",
                        1);
  arg_parser.add_option(NUM_TEST_SAMPLES,
                        {"--num_test_samples"},
                        utils::ENV("LBANN_NUM_TEST_SAMPLES"),
                        "Set the number of testing samples to ingest.",
                        -1);
  arg_parser.add_option(NUM_TRAIN_SAMPLES,
                        {"--num_train_samples"},
                        utils::ENV("LBANN_NUM_TRAIN_SAMPLES"),
                        "Set the number of training samples to ingest.",
                        -1);
  arg_parser.add_option(NUM_VALIDATE_SAMPLES,
                        {"--num_validate_samples"},
                        utils::ENV("LBANN_NUM_VALIDATE_SAMPLES"),
                        "Set the number of validate samples to ingest.",
                        -1);
  arg_parser.add_option(OPTIMIZER,
                        {"--optimizer"},
                        "TODO",
                        "");
  arg_parser.add_option(PROCS_PER_TRAINER,
                        {"--procs_per_trainer"},
                        utils::ENV("LBANN_PROCS_PER_TRAINER"),
                        "Number of MPI ranks per LBANN trainer, "
                        "If the field is not set (or set to 0) then "
                        " all MPI ranks are assigned to one trainer."
                        " The number of processes per trainer must "
                        " evenly divide the total number of MPI ranks. "
                        " The number of resulting trainers is "
                        " num_procs / procs_per_trainer.",
                        -1);
  arg_parser.add_option(PROTOTEXT,
                        {"--prototext"},
                        "Prototext file containing experiment",
                        "");
  arg_parser.add_option(RANDOM_SEED,
                        {"--random_seed", "--rand_seed"},
                        "Value to seed RNG",
                        -1);
  arg_parser.add_option(READER,
                        {"--reader"},
                        "TODO",
                        "");
  arg_parser.add_option(RESTART_DIR,
                        {"--restart_dir"},
                        "Restart from a checkpoint found in the given directory.\n"
                        "If the directory doesn't exist or doesn't contain a checkpoint,\n"
                        "an error will be thrown.\n",
                        "");
  arg_parser.add_option(TRAINER_CREATE_TWO_MODELS,
                        {"--trainer_create_two_models"},
                        utils::ENV("LBANN_TRAINER_CREATE_TWO_MODELS"),
                        "Create two models (one each for primary and secondary grid). "
                        "Default is False.",
                        false);
  arg_parser.add_option(TRAINER_GRID_HEIGHT,
                        {"--trainer_grid_height"},
                        utils::ENV("LBANN_TRAINER_GRID_HEIGHT"),
                        "Height of 2D process grid for each trainer. "
                        "Default grid is approximately square.",
                        -1);
  arg_parser.add_option(TRAINER_PRIMARY_GRID_SIZE,
                        {"--trainer_primary_grid_size"},
                        utils::ENV("LBANN_TRAINER_PRIMARY_GRID_SIZE"),
                        "Primary grid size per trainer. "
                        "Disables Sub-grid parallelism, when it is 0",
                        0);


  // Unused (?) options
  arg_parser.add_option(CHECKPOINT_DIR,
                        {"--checkpoint_dir"},
                        "Save to or restart from a specific checkpoint directory.\n"
                        "Additionally, sets the output directory for dumping weights.\n"
                        "Modifies callbacks: checkpoint, save_model, dump_weights\n",
                        "");
  arg_parser.add_option(DATA_LAYOUT,
                        {"--data_layout"},
                        "must be: data_parallel or model_parallel\n"
                        "note: this will be applied to all layers, metrics (and others)\n"
                        "that take DATA_PARALLEL or MODEL_PARALLEL as a template parameter",
                        "data_parallel");
  arg_parser.add_option(OBJECTIVE_FUNCTION,
                        {"--objective_function"},
                        "must be: categorical_cross_entropy or mean_squared_error",
                        "categorical_cross_entropy");
  arg_parser.add_option(SMILES_BUFFER_SIZE,
                        {"--smiles_buffer_size"},
                        utils::ENV("LBANN_SMILES_BUFFER_SIZE"),
                        "Size of the read buffer for the SMILES "
                        "data reader.",
                        16*1024*1024UL);
  arg_parser.add_flag(SUPER_NODE,
                      {"--super_node"},
                      "Enables the data store in-memory structure to use the supernode exchange structure");
}

void construct_datastore_options() {
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(DATA_STORE_CACHE,
                      {"--data_store_cache"},
                      "TODO");
  arg_parser.add_flag(DATA_STORE_DEBUG,
                      {"--data_store_debug"},
                      "TODO");
  arg_parser.add_flag(DATA_STORE_FAIL,
                      {"--data_store_fail"},
                      "TODO");
  arg_parser.add_flag(DATA_STORE_MIN_MAX_TIMING,
                      {"--data_store_min_max_timing"},
                      "TODO");
  arg_parser.add_flag(DATA_STORE_NO_THREAD,
                      {"--data_store_no_thread"},
                      "TODO");
  arg_parser.add_flag(DATA_STORE_PROFILE,
                      {"--data_store_profile"},
                      "TODO");
  arg_parser.add_flag(DATA_STORE_TEST_CACHE,
                      {"--data_store_test_cache"},
                      "TODO");

  // Input options
  arg_parser.add_option(DATA_STORE_SPILL,
                        {"--data_store_spill"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_STORE_TEST_CHECKPOINT,
                        {"--data_store_test_checkpoint"},
                        "TODO",
                        "");
}

void construct_datareader_options() {
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(ALL_GATHER_OLD,
                      {"--all_gather_old"},
                      "TODO");
  arg_parser.add_flag(CHECK_DATA,
                      {"--check_data"},
                      "TODO");
  arg_parser.add_flag(CREATE_TARBALL,
                      {"--create_tarball"},
                      "TODO");
  arg_parser.add_flag(DEBUG_CONCATENATE,
                      {"--debug_concatenate"},
                      "TODO");
  arg_parser.add_flag(DISABLE_SIGNAL_HANDLER,
                      {"--disable_signal_handler"},
                      "TODO");
  arg_parser.add_flag(EXIT_AFTER_SETUP,
                      {"--exit_after_setup"},
                      "TODO");
  arg_parser.add_flag(GENERATE_MULTI_PROTO,
                      {"--generate_multi_proto"},
                      "TODO");
  arg_parser.add_flag(KEEP_SAMPLE_ORDER,
                      {"--keep_sample_order"},
                      "TODO");
  arg_parser.add_flag(KEEP_PACKED_FIELDS,
                      {"--keep_packed_fields"},
                      "TODO");
  arg_parser.add_flag(LOAD_FULL_SAMPLE_LIST_ONCE,
                      {"--load_full_sample_list_once"},
                      "TODO");
  arg_parser.add_flag(MAKE_TEST_FAIL,
                      {"--make_test_fail"},
                      "TODO");
  arg_parser.add_flag(NODE_SIZES_VARY,
                      {"--node_sizes_vary"},
                      "TODO");
  arg_parser.add_flag(QUIET,
                      {"--quiet"},
                      "TODO");
  arg_parser.add_flag(STACK_TRACE_TO_FILE,
                      {"--stack_trace_to_file"},
                      "TODO");
  arg_parser.add_flag(TEST_ENCODE,
                      {"--test_encode"},
                      "TODO");
  arg_parser.add_flag(Z_SCORE,
                      {"--z_score"},
                      "TODO");

  // Input options
  arg_parser.add_option(ABSOLUTE_SAMPLE_COUNT,
                        {"--absolute_sample_count"},
                        "TODO",
                        -1);
  arg_parser.add_option(DATA_FILEDIR,
                        {"--data_filedir"},
                        "Sets the file direcotry for train and test data",
                        "");
  arg_parser.add_option(DATA_FILEDIR_TEST,
                        {"--data_filedir_test"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_FILEDIR_TRAIN,
                        {"--data_filedir_train"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_FILEDIR_VALIDATE,
                        {"--data_filedir_validate"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_FILENAME_TEST,
                        {"--data_filename_test"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_FILENAME_TRAIN,
                        {"--data_filename_train"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_FILENAME_VALIDATE,
                        {"--data_filename_validate"},
                        "TODO",
                        "");
  arg_parser.add_option(DATA_READER_PERCENT,
                        {"--data_reader_percent"},
                        "TODO",
                        (float)-1);
  arg_parser.add_option(DELIMITER,
                        {"--delimiter"},
                        "TODO",
                        "");
  arg_parser.add_option(IMAGE_SIZES_FILENAME,
                        {"--image_sizes_filename"},
                        "TODO",
                        "");
  arg_parser.add_option(LABEL_FILENAME_TEST,
                        {"--label_filename_test"},
                        "TODO",
                        "");
  arg_parser.add_option(LABEL_FILENAME_TRAIN,
                        {"--label_filename_train"},
                        "TODO",
                        "");
  arg_parser.add_option(LABEL_FILENAME_VALIDATE,
                        {"--label_filename_validate"},
                        "TODO",
                        "");
  arg_parser.add_option(NORMALIZATION,
                        {"--normalization"},
                        "TODO",
                        "");
  arg_parser.add_option(N_LINES,
                        {"--n_lines"},
                        "TODO",
                        -1);
  arg_parser.add_option(PAD_INDEX,
                        {"--pad_index"},
                        "TODO",
                        -1);
  arg_parser.add_option(PILOT2_READ_FILE_SIZES,
                        {"--pilot2_read_file_sizes"},
                        "TODO",
                        "");
  arg_parser.add_option(PILOT2_SAVE_FILE_SIZES,
                        {"--pilot2_save_file_sizes"},
                        "TODO",
                        "");
  arg_parser.add_option(SAMPLE_LIST_TEST,
                        {"--sample_list_test"},
                        "TODO",
                        "");
  arg_parser.add_option(SAMPLE_LIST_TRAIN,
                        {"--sample_list_train"},
                        "TODO",
                        "");
  arg_parser.add_option(SAMPLE_LIST_VALIDATE,
                        {"--sample_list_validate"},
                        "TODO",
                        "");
  arg_parser.add_option(SEQUENCE_LENGTH,
                        {"--sequence_length", "--seq_len"},
                        "TODO",
                        -1);
  arg_parser.add_option(TEST_TARBALL,
                        {"--test_tarball"},
                        "TODO",
                        -1);
  arg_parser.add_option(VOCAB,
                        {"--vocab"},
                        "TODO",
                        "");

  // Unused (?) options
  arg_parser.add_flag(SHARE_TESTING_DATA_READERS,
                      {"--share_testing_data_readers"},
                      "TODO");
  arg_parser.add_flag(WRITE_SAMPLE_LABEL_LIST,
                      {"--write_sample_label_list"},
                      "TODO");
}

void construct_jag_options() {
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(JAG,
                      {"--jag"},
                      "TODO");
  arg_parser.add_flag(JAG_PARTITIONED,
                      {"--jag_partitioned"},
                      "TODO");

  // Input options
  arg_parser.add_option(BASE_DIR,
                        {"--base_dir"},
                        "TODO",
                        "");
  arg_parser.add_option(FILELIST,
                        {"--filelist"},
                        "TODO",
                        "");
  arg_parser.add_option(FILENAME,
                        {"--filename"},
                        "TODO",
                        "");
  arg_parser.add_option(FORMAT,
                        {"--format"},
                        "TODO",
                        "");
  arg_parser.add_option(INDEX_FN,
                        {"--index_fn"},
                        "TODO",
                        "");
  arg_parser.add_option(MAPPING_FN,
                        {"--mapping_fn"},
                        "TODO",
                        "");
  arg_parser.add_option(NUM_LISTS,
                        {"--num_lists"},
                        "TODO",
                        -1);
  arg_parser.add_option(NUM_SAMPLES,
                        {"--num_samples"},
                        "TODO",
                        -1);
  arg_parser.add_option(NUM_SAMPLES_PER_FILE,
                        {"--num_samples_per_file"},
                        "TODO",
                        -1);
  arg_parser.add_option(NUM_SAMPLES_PER_LIST,
                        {"--num_samples_per_list"},
                        "TODO",
                        -1);
  arg_parser.add_option(NUM_SUBDIRS,
                        {"--num_subdirs"},
                        "TODO",
                        -1);
  arg_parser.add_option(OUTPUT_BASE_DIR,
                        {"--output_base_dir"},
                        "TODO",
                        "");
  arg_parser.add_option(OUTPUT_BASE_FN,
                        {"--output_base_fn"},
                        "TODO",
                        "");
  arg_parser.add_option(OUTPUT_DIR,
                        {"--output_dir"},
                        "TODO",
                        "");
  arg_parser.add_option(OUTPUT_FN,
                        {"--output_fn"},
                        "TODO",
                        "");
  arg_parser.add_option(SAMPLES_PER_FILE,
                        {"--samples_per_file"},
                        "TODO",
                        -1);
}

void construct_callback_options() {
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(NO_IM_COMM,
                      {"--no_im_comm"},
                      "removed ImComm callback, if present; this is intended for"
                      "running alexnet with a single model, but may be useful elsewhere");

  // Input options
  arg_parser.add_option(CKPT_DIR,
                        {"--ckpt_dir"},
                        "TODO",
                        "");

  // Unused (?) options
  arg_parser.add_option(IMAGE_DIR,
                        {"--image_dir"},
                        "if the model has callback_save_images, this determines where the"
                        "images are saved",
                        "");
}

void construct_all_options() {
  construct_std_options();
  construct_datastore_options();
  construct_datareader_options();
  construct_jag_options();
  construct_callback_options();
}

} // namespace lbann
