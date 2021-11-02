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

void construct_std_options()
{
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(
    LBANN_OPTION_DISABLE_BACKGROUND_IO_ACTIVITY,
    {"--disable_background_io_activity"},
    "[STD] prevent the input layers from fetching data in the background");
  arg_parser.add_flag(
    LBANN_OPTION_DISABLE_CUDA,
    {"--disable_cuda"},
    "[STD] has no effect unless LBANN was compiled with LBANN_HAS_CUDNN");
  arg_parser.add_flag(
    LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE,
    {"--load_model_weights_dir_is_complete"},
    "[STD] Use load_model_weights_dir as given, ignoring checkpoint hierarchy");
  arg_parser.add_flag(LBANN_OPTION_LTFB_ALLOW_GLOBAL_STATISTICS,
                      {"--ltfb_allow_global_statistics"},
                      utils::ENV("LBANN_LTFB_ALLOW_GLOBAL_STATISTICS"),
                      "[STD] Allow the print_statistics callback to report "
                      "global (inter-trainer) summary statistics.");
  arg_parser.add_flag(
    LBANN_OPTION_LTFB_VERBOSE,
    {"--ltfb_verbose"},
    "[STD] Increases number of per-trainer messages that are reported");
  arg_parser.add_flag(
    LBANN_OPTION_NO_IM_COMM,
    {"--no_im_comm"},
    "[STD] removed ImComm callback, if present; this is intended for"
    "running alexnet with a single model, but may be useful elsewhere");
  arg_parser.add_flag(LBANN_OPTION_PRELOAD_DATA_STORE,
                      {"--preload_data_store"},
                      "[STD] Preloads the data store in-memory structure "
                      "druing data reader load time");
  arg_parser.add_flag(
    LBANN_OPTION_PRINT_AFFINITY,
    {"--print_affinity"},
    "[STD] display information on how OpenMP threads are provisioned");
  arg_parser.add_flag(
    LBANN_OPTION_SERIALIZE_IO,
    {"--serialize_io"},
    "[STD] force data readers to use a single threaded for I/O");
  arg_parser.add_flag(LBANN_OPTION_ST_FULL_TRACE, {"--st_full_trace"}, "[STD] TODO");
  arg_parser.add_flag(LBANN_OPTION_ST_ON, {"--st_on"}, "[STD] TODO");
  arg_parser.add_flag(LBANN_OPTION_USE_CUBLAS_TENSOR_OPS,
                      {"--use-cublas-tensor-ops"},
                      utils::ENV("LBANN_USE_CUBLAS_TENSOR_OPS"),
                      "[STD] Set the default cuBLAS math mode to use "
                      "Tensor Core operations when available.");
  arg_parser.add_flag(LBANN_OPTION_USE_CUDNN_TENSOR_OPS,
                      {"--use-cudnn-tensor-ops"},
                      utils::ENV("LBANN_USE_CUDNN_TENSOR_OPS"),
                      "[STD] Set the default cuDNN math mode to use "
                      "Tensor Core operations when available.");
  arg_parser.add_flag(LBANN_OPTION_USE_DATA_STORE,
                      {"--use_data_store"},
                      "[STD] Enables the data store in-memory structure");
  arg_parser.add_flag(LBANN_OPTION_USE_LTFB, {"--ltfb"}, "[STD] TODO");
  arg_parser.add_flag(LBANN_OPTION_VERBOSE,
                      {"--verbose", "--verbose_print"},
                      "[STD] Turns on verbose mode");
  arg_parser.add_flag(LBANN_OPTION_WRITE_SAMPLE_LIST,
                      {"--write_sample_list"},
                      "[STD] Writes out the sample list that was loaded into "
                      "the current directory");
  arg_parser.add_flag(
    LBANN_OPTION_USE_GPU_DEFAULT_MEMORY_IN_FORWARD_PROP,
    {"--use_gpu_default_memory_in_forward_prop"},
    utils::ENV("LBANN_GPU_DEFAULT_MEMORY_IN_FORWARD_PROP"),
    "[STD] Use Hydrogen's default memory mode for GPU buffers in "
    "forward prop (namely activations and weights). This will "
    "typically use a GPU memory pool, which uses more memory than "
    "directly allocating GPU memory.");
  arg_parser.add_flag(
    LBANN_OPTION_INIT_SHMEM,
    {"--init_shmem"},
    utils::ENV("LBANN_INIT_SHMEM"),
    "[STD] Initialize SHMEM when initializing LBANN");
  arg_parser.add_flag(
    LBANN_OPTION_INIT_NVSHMEM,
    {"--init_nvshmem"},
    utils::ENV("LBANN_INIT_NVSHMEM"),
    "[STD] Initialize NVSHMEM when initializing LBANN");

  // Input options
  arg_parser.add_option(
    LBANN_OPTION_CKPT_DIR,
    {"--checkpoint_dir", "--ckpt_dir"},
    "[STD] Save to or restart from a specific checkpoint directory.\n"
    "Additionally, sets the output directory for dumping weights.\n"
    "Modifies callbacks: checkpoint, save_model, dump_weights\n",
    "");
  arg_parser.add_option(LBANN_OPTION_HYDROGEN_BLOCK_SIZE,
                        {"--hydrogen_block_size"},
                        "[STD] Block size for Hydrogen",
                        0);
  arg_parser.add_option(
    LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR,
    {"--load_model_weights_dir"},
    "[STD] Load model wieghts found in the given directory.\n"
    "If the directory doesn't exist, doesn't contain valid weights,\n"
    "or doesn't contain a checkpoint,\n"
    "an error will be thrown.\n",
    "");
  arg_parser.add_option(
    LBANN_OPTION_MAX_RNG_SEEDS_DISPLAY,
    {"--rng_seeds_per_trainer_to_display"},
    utils::ENV("LBANN_RNG_SEEDS_PER_TRAINER_TO_DISPLAY"),
    "[STD] Limit how many random seeds LBANN should display "
    "from each trainer",
    2);
  arg_parser.add_option(LBANN_OPTION_METADATA, {"--metadata"}, "[STD] TODO", "");
  arg_parser.add_option(LBANN_OPTION_MINI_BATCH_SIZE,
                        {"--mini_batch_size"},
                        "[STD] Size of mini batches",
                        -1);
  arg_parser.add_option(LBANN_OPTION_MODEL, {"--model"}, "[STD] TODO", "");
  arg_parser.add_option(LBANN_OPTION_NUM_EPOCHS,
                        {"--num_epochs"},
                        "[STD] Number of epochs to train model",
                        -1);
  arg_parser.add_option(LBANN_OPTION_NUM_IO_THREADS,
                        {"--num_io_threads"},
                        utils::ENV("LBANN_NUM_IO_THREADS"),
                        "[STD] Number of threads available to both I/O and "
                        "initial data transformations for each rank.",
                        64);
  arg_parser.add_option(LBANN_OPTION_NUM_PARALLEL_READERS,
                        {"--num_parallel_readers"},
                        "[STD] The number of parallel data readers",
                        1);
  arg_parser.add_option(LBANN_OPTION_NUM_TEST_SAMPLES,
                        {"--num_test_samples"},
                        utils::ENV("LBANN_NUM_TEST_SAMPLES"),
                        "[STD] Set the number of testing samples to ingest.",
                        -1);
  arg_parser.add_option(LBANN_OPTION_NUM_TRAIN_SAMPLES,
                        {"--num_train_samples"},
                        utils::ENV("LBANN_NUM_TRAIN_SAMPLES"),
                        "[STD] Set the number of training samples to ingest.",
                        -1);
  arg_parser.add_option(LBANN_OPTION_NUM_VALIDATE_SAMPLES,
                        {"--num_validate_samples"},
                        utils::ENV("LBANN_NUM_VALIDATE_SAMPLES"),
                        "[STD] Set the number of validate samples to ingest.",
                        -1);
  arg_parser.add_option(LBANN_OPTION_OPTIMIZER, {"--optimizer"}, "[STD] TODO", "");
  arg_parser.add_option(LBANN_OPTION_PROCS_PER_TRAINER,
                        {"--procs_per_trainer"},
                        utils::ENV("LBANN_PROCS_PER_TRAINER"),
                        "[STD] Number of MPI ranks per LBANN trainer, "
                        "If the field is not set (or set to 0) then "
                        " all MPI ranks are assigned to one trainer."
                        " The number of processes per trainer must "
                        " evenly divide the total number of MPI ranks. "
                        " The number of resulting trainers is "
                        " num_procs / procs_per_trainer.",
                        -1);
  arg_parser.add_option(LBANN_OPTION_PROTOTEXT,
                        {"--prototext"},
                        "[STD] Prototext file containing experiment",
                        "");
  arg_parser.add_option(LBANN_OPTION_RANDOM_SEED,
                        {"--random_seed", "--rand_seed"},
                        utils::ENV("LBANN_RANDOM_SEED"),
                        "[STD] RNG seed",
                        0);
  arg_parser.add_option(LBANN_OPTION_READER, {"--reader"}, "[STD] TODO", "");
  arg_parser.add_option(
    LBANN_OPTION_RESTART_DIR,
    {"--restart_dir"},
    "[STD] Restart from a checkpoint found in the given directory.\n"
    "If the directory doesn't exist or doesn't contain a checkpoint,\n"
    "an error will be thrown.\n",
    "");
  arg_parser.add_option(
    LBANN_OPTION_TRAINER_CREATE_TWO_MODELS,
    {"--trainer_create_two_models"},
    utils::ENV("LBANN_TRAINER_CREATE_TWO_MODELS"),
    "[STD] Create two models (one each for primary and secondary grid). "
    "Default is False.",
    false);
  arg_parser.add_option(LBANN_OPTION_TRAINER_GRID_HEIGHT,
                        {"--trainer_grid_height"},
                        utils::ENV("LBANN_TRAINER_GRID_HEIGHT"),
                        "[STD] Height of 2D process grid for each trainer. "
                        "Default grid is approximately square.",
                        -1);
  arg_parser.add_option(LBANN_OPTION_TRAINER_PRIMARY_GRID_SIZE,
                        {"--trainer_primary_grid_size"},
                        utils::ENV("LBANN_TRAINER_PRIMARY_GRID_SIZE"),
                        "[STD] Primary grid size per trainer. "
                        "Disables Sub-grid parallelism, when it is 0",
                        0);
}

void construct_datastore_options()
{
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_CACHE,
                      {"--data_store_cache"},
                      "[DATASTORE] TODO");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_DEBUG,
                      {"--data_store_debug"},
                      "[DATASTORE] TODO");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_FAIL,
                      {"--data_store_fail"},
                      "[DATASTORE] TODO");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_MIN_MAX_TIMING,
                      {"--data_store_min_max_timing"},
                      "[DATASTORE] TODO");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_NO_THREAD,
                      {"--data_store_no_thread"},
                      "[DATASTORE] TODO");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_PROFILE,
                      {"--data_store_profile"},
                      "[DATASTORE] TODO");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_TEST_CACHE,
                      {"--data_store_test_cache"},
                      "[DATASTORE] TODO");

  // Input options
  arg_parser.add_option(LBANN_OPTION_DATA_STORE_SPILL,
                        {"--data_store_spill"},
                        "[DATASTORE] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_STORE_TEST_CHECKPOINT,
                        {"--data_store_test_checkpoint"},
                        "[DATASTORE] TODO",
                        "");
}

void construct_datareader_options()
{
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(LBANN_OPTION_ALL_GATHER_OLD,
                      {"--all_gather_old"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_CHECK_DATA, {"--check_data"}, "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_CREATE_TARBALL,
                      {"--create_tarball"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_DEBUG_CONCATENATE,
                      {"--debug_concatenate"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_DISABLE_SIGNAL_HANDLER,
                      {"--disable_signal_handler"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_EXIT_AFTER_SETUP,
                      {"--exit_after_setup"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_GENERATE_MULTI_PROTO,
                      {"--generate_multi_proto"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_KEEP_SAMPLE_ORDER,
                      {"--keep_sample_order"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_KEEP_PACKED_FIELDS,
                      {"--keep_packed_fields"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_LOAD_FULL_SAMPLE_LIST_ONCE,
                      {"--load_full_sample_list_once"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_MAKE_TEST_FAIL,
                      {"--make_test_fail"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_NODE_SIZES_VARY,
                      {"--node_sizes_vary"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_QUIET, {"--quiet"}, "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_STACK_TRACE_TO_FILE,
                      {"--stack_trace_to_file"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_TEST_ENCODE, {"--test_encode"}, "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_WRITE_SAMPLE_LABEL_LIST,
                      {"--write_sample_label_list"},
                      "[DATAREADER] TODO");
  arg_parser.add_flag(LBANN_OPTION_Z_SCORE, {"--z_score"}, "[DATAREADER] TODO");

  // Input options
  arg_parser.add_option(LBANN_OPTION_ABSOLUTE_SAMPLE_COUNT,
                        {"--absolute_sample_count"},
                        "[DATAREADER] TODO",
                        -1);
  arg_parser.add_option(
    LBANN_OPTION_DATA_FILEDIR,
    {"--data_filedir"},
    "[DATAREADER] Sets the file direcotry for train and test data",
    "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILEDIR_TEST,
                        {"--data_filedir_test"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILEDIR_TRAIN,
                        {"--data_filedir_train"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILEDIR_VALIDATE,
                        {"--data_filedir_validate"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILENAME_TEST,
                        {"--data_filename_test"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILENAME_TRAIN,
                        {"--data_filename_train"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILENAME_VALIDATE,
                        {"--data_filename_validate"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_READER_PERCENT,
                        {"--data_reader_percent"},
                        "[DATAREADER] TODO",
                        (float)-1);
  arg_parser.add_option(LBANN_OPTION_DELIMITER, {"--delimiter"}, "[DATAREADER] TODO", "");
  arg_parser.add_option(LBANN_OPTION_IMAGE_SIZES_FILENAME,
                        {"--image_sizes_filename"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_LABEL_FILENAME_TEST,
                        {"--label_filename_test"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_LABEL_FILENAME_TRAIN,
                        {"--label_filename_train"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_LABEL_FILENAME_VALIDATE,
                        {"--label_filename_validate"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_NORMALIZATION,
                        {"--normalization"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_N_LINES, {"--n_lines"}, "[DATAREADER] TODO", -1);
  arg_parser.add_option(LBANN_OPTION_PAD_INDEX, {"--pad_index"}, "[DATAREADER] TODO", -1);
  arg_parser.add_option(LBANN_OPTION_PILOT2_READ_FILE_SIZES,
                        {"--pilot2_read_file_sizes"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_PILOT2_SAVE_FILE_SIZES,
                        {"--pilot2_save_file_sizes"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_SAMPLE_LIST_TEST,
                        {"--sample_list_test"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_SAMPLE_LIST_TRAIN,
                        {"--sample_list_train"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_SAMPLE_LIST_VALIDATE,
                        {"--sample_list_validate"},
                        "[DATAREADER] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_SEQUENCE_LENGTH,
                        {"--sequence_length", "--seq_len"},
                        "[DATAREADER] TODO",
                        -1);
  arg_parser.add_option(LBANN_OPTION_SMILES_BUFFER_SIZE,
                        {"--smiles_buffer_size"},
                        utils::ENV("LBANN_SMILES_BUFFER_SIZE"),
                        "[DATAREADER] Size of the read buffer for the SMILES "
                        "data reader.",
                        16 * 1024 * 1024UL);
  arg_parser.add_option(LBANN_OPTION_TEST_TARBALL,
                        {"--test_tarball"},
                        "[DATAREADER] TODO",
                        -1);
  arg_parser.add_option(LBANN_OPTION_VOCAB, {"--vocab"}, "[DATAREADER] TODO", "");
}

void construct_jag_options()
{
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(LBANN_OPTION_JAG, {"--jag"}, "[JAG] TODO");
  arg_parser.add_flag(LBANN_OPTION_JAG_PARTITIONED, {"--jag_partitioned"}, "[JAG] TODO");

  // Input options
  arg_parser.add_option(LBANN_OPTION_BASE_DIR, {"--base_dir"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_FILELIST, {"--filelist"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_FILENAME, {"--filename"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_FORMAT, {"--format"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_INDEX_FN, {"--index_fn"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_MAPPING_FN, {"--mapping_fn"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_NUM_LISTS, {"--num_lists"}, "[JAG] TODO", -1);
  arg_parser.add_option(LBANN_OPTION_NUM_SAMPLES, {"--num_samples"}, "[JAG] TODO", -1);
  arg_parser.add_option(LBANN_OPTION_NUM_SAMPLES_PER_FILE,
                        {"--num_samples_per_file"},
                        "[JAG] TODO",
                        1000);
  arg_parser.add_option(LBANN_OPTION_NUM_SAMPLES_PER_LIST,
                        {"--num_samples_per_list"},
                        "[JAG] TODO",
                        -1);
  arg_parser.add_option(LBANN_OPTION_NUM_SUBDIRS, {"--num_subdirs"}, "[JAG] TODO", -1);
  arg_parser.add_option(LBANN_OPTION_OUTPUT_BASE_DIR,
                        {"--output_base_dir"},
                        "[JAG] TODO",
                        "");
  arg_parser.add_option(LBANN_OPTION_OUTPUT_BASE_FN, {"--output_base_fn"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_OUTPUT_DIR, {"--output_dir"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_OUTPUT_FN, {"--output_fn"}, "[JAG] TODO", "");
  arg_parser.add_option(LBANN_OPTION_SAMPLES_PER_FILE,
                        {"--samples_per_file"},
                        "[JAG] TODO",
                        -1);
}

void construct_all_options()
{
  construct_std_options();
  construct_datastore_options();
  construct_datareader_options();
  construct_jag_options();
}

} // namespace lbann
