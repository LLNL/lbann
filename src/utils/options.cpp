////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include "lbann_config.hpp"

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
    LBANN_OPTION_DISABLE_DISTCONV,
    {"--disable_distconv"},
    utils::ENV("LBANN_DISABLE_DISTCONV"),
    "[STD] Disables distconv support. Has no effect unless LBANN was compiled "
    "with LBANN_HAS_DISTCONV");
  arg_parser.add_flag(
    LBANN_OPTION_DISABLE_SIGNAL_HANDLER,
    {"--disable_signal_handler"},
    "[STD] Disables signal handling (signal handling on by default)");
  arg_parser.add_flag(LBANN_OPTION_EXIT_AFTER_SETUP,
                      {"--exit_after_setup"},
                      "[STD] Forces exit after model setup");
  arg_parser.add_flag(LBANN_OPTION_GENERATE_MULTI_PROTO,
                      {"--generate_multi_proto"},
                      "[STD] Enables loading of multiple prototext files for "
                      "model, datareader, optimizer, etc. input options");
  arg_parser.add_flag(
    LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE,
    {"--load_model_weights_dir_is_complete"},
    "[STD] Use load_model_weights_dir as given, ignoring checkpoint hierarchy");
  arg_parser.add_flag(
    LBANN_OPTION_LTFB_ALLOW_GLOBAL_STATISTICS,
    {"--ltfb_allow_global_statistics"},
    utils::ENV("LBANN_LTFB_ALLOW_GLOBAL_STATISTICS"),
    "[STD, deprecated] Allow the print_statistics callback to report "
    "global (inter-trainer) summary statistics.");
  arg_parser.add_flag(LBANN_OPTION_LTFB_VERBOSE,
                      {"--ltfb_verbose"},
                      utils::ENV("LBANN_LTFB_VERBOSE"),
                      "[STD, deprecated] Increases number of per-trainer "
                      "messages that are reported");
  arg_parser.add_flag(LBANN_OPTION_ALLOW_MULTITRAINER_GLOBAL_STATISTICS,
                      {"--ltfb_global_multitrainer_statistics"},
                      utils::ENV("LBANN_ALLOW_MULTITRAINER_GLOBAL_STATISTICS"),
                      "[STD] Allow the print_statistics callback to report "
                      "global (inter-trainer) summary statistics.");
  arg_parser.add_flag(
    LBANN_OPTION_MULTITRAINER_VERBOSE,
    {"--multitrainer_verbose"},
    utils::ENV("LBANN_MULTITRAINER_VERBOSE"),
    "[STD] Increases number of per-trainer messages that are reported");
  arg_parser.add_flag(LBANN_OPTION_PRELOAD_DATA_STORE,
                      {"--preload_data_store"},
                      "[STD] Preloads the data store in-memory structure "
                      "during data reader load time");
  arg_parser.add_flag(
    LBANN_OPTION_PRINT_AFFINITY,
    {"--print_affinity"},
    "[STD] display information on how OpenMP threads are provisioned");
  arg_parser.add_flag(
    LBANN_OPTION_SERIALIZE_IO,
    {"--serialize_io"},
    "[STD] force data readers to use a single threaded for I/O");
  arg_parser.add_flag(LBANN_OPTION_STACK_TRACE_TO_FILE,
                      {"--stack_trace_to_file"},
                      "[STD] When enabled, stack trace is output to file");
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
  arg_parser.add_flag(LBANN_OPTION_VERBOSE,
                      {"--verbose", "--verbose_print"},
                      "[STD] Turns on verbose mode");
  arg_parser.add_flag(
    LBANN_OPTION_USE_GPU_DIRECT_MEMORY_IN_FORWARD_PROP,
    {"--use_gpu_direct_memory_in_forward_prop"},
    utils::ENV("LBANN_GPU_DIRECT_MEMORY_IN_FORWARD_PROP"),
    "[STD] Use direct memory mode (i.e., non-pooled) for GPU buffers in "
    "forward prop (namely activations and weights). "
    "The GPU memory pool typically uses more memory than "
    "directly allocating GPU memory.");
  arg_parser.add_flag(LBANN_OPTION_INIT_SHMEM,
                      {"--init_shmem"},
                      utils::ENV("LBANN_INIT_SHMEM"),
                      "[STD] Initialize SHMEM when initializing LBANN");
  arg_parser.add_flag(LBANN_OPTION_INIT_NVSHMEM,
                      {"--init_nvshmem"},
                      utils::ENV("LBANN_INIT_NVSHMEM"),
                      "[STD] Initialize NVSHMEM when initializing LBANN");
  arg_parser.add_flag(LBANN_OPTION_NO_INPLACE,
                      {"--no_inplace"},
                      utils::ENV("LBANN_NO_INPLACE"),
                      "[STD] Disable in-place layer memory optimization");
  arg_parser.add_flag(LBANN_OPTION_NO_BACKPROP_DISABLE,
                      {"--no_backprop_disable"},
                      utils::ENV("LBANN_NO_BACKPROP_DISABLE"),
                      "[STD] Always compute all layers in backpropagation");

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
    "[STD] Load model weights found in the given directory.\n"
    "If the directory doesn't exist, doesn't contain valid weights,\n"
    "or doesn't contain a checkpoint, an error will be thrown.\n",
    "");
  arg_parser.add_option(
    LBANN_OPTION_MAX_RNG_SEEDS_DISPLAY,
    {"--rng_seeds_per_trainer_to_display"},
    utils::ENV("LBANN_RNG_SEEDS_PER_TRAINER_TO_DISPLAY"),
    "[STD] Limit how many random seeds LBANN should display "
    "from each trainer",
    2);
  arg_parser.add_option(LBANN_OPTION_METADATA,
                        {"--metadata"},
                        "[STD] Metadata input file",
                        "");
  arg_parser.add_option(LBANN_OPTION_MINI_BATCH_SIZE,
                        {"--mini_batch_size"},
                        "[STD] Size of mini batches",
                        -1);
  arg_parser.add_option(LBANN_OPTION_MODEL,
                        {"--model"},
                        "[STD] Model input file",
                        "");
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
  arg_parser.add_option(LBANN_OPTION_OPTIMIZER,
                        {"--optimizer"},
                        "[STD] Optimizer input file",
                        "");
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
  arg_parser.add_option(LBANN_OPTION_READER,
                        {"--reader"},
                        "[STD] Data reader input file",
                        "");
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
  arg_parser.add_option(
    LBANN_OPTION_TRAINER_ENABLE_SUBGRID_ASYNC_COMM,
    {"--trainer_enable_subgrid_async_comm"},
    utils::ENV("LBANN_TRAINER_ENABLE_SUBGRID_ASYNC_COMM"),
    "Enable asynchronous communication in sub-grid parallelism. "
    "Default is False.",
    false);
  arg_parser.add_option(
    LBANN_OPTION_TRAINER_ENABLE_TOPO_AWARE_SUBGRID,
    {"--trainer_enable_topo_aware_subgrid"},
    utils::ENV("LBANN_TRAINER_ENABLE_TOPO_AWARE_SUBGRID"),
    "Enable topology aware process placement in sub-grid parallelism. "
    "Default is False.",
    false);
  arg_parser.add_option(LBANN_OPTION_NUM_SUBGRIDS_BLOCK_ORDER,
                        {"--num-subgrids", "--num-subgrids-block-order"},
                        utils::ENV("LBANN_NUM_SUBGRIDS"),
                        "[STD] Divide each trainer into equally-sized "
                        "sub-grids with blocked ordering",
                        0);
#ifdef LBANN_HAS_CALIPER
  arg_parser.add_flag(LBANN_OPTION_USE_CALIPER,
                      {"--caliper"},
                      "[STD] Enable caliper.");
  arg_parser.add_option(LBANN_OPTION_CALIPER_CONFIG,
                        {"--caliper_config"},
                        "[STD] Caliper configuration string",
                        "spot");
#endif
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
                      "[DATASTORE] Enables data store debug output for each "
                      "<rank, reader_role> pair");
  arg_parser.add_flag(
    LBANN_OPTION_DATA_STORE_FAIL,
    {"--data_store_fail"},
    "[DATASTORE] Forces data store to fail, used for testing purposes");
  arg_parser.add_flag(
    LBANN_OPTION_DATA_STORE_MIN_MAX_TIMING,
    {"--data_store_min_max_timing"},
    "[DATASTORE] Enables data store min and max times output to profile for "
    "various data store operations, data store profiling must be enabled");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_NO_THREAD,
                      {"--data_store_no_thread"},
                      "[DATASTORE] Disables data store I/O multi-threading");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_PROFILE,
                      {"--data_store_profile"},
                      "[DATASTORE] Enable data store profiling output for each "
                      "<P_0, reader_role> pair");
  arg_parser.add_flag(LBANN_OPTION_DATA_STORE_TEST_CACHE,
                      {"--data_store_test_cache"},
                      "[DATASTORE] Perform checks on imagenet data store "
                      "cache, used for testing purposes");
  arg_parser.add_flag(
    LBANN_OPTION_NODE_SIZES_VARY,
    {"--node_sizes_vary"},
    "[DATASTORE] Allows Conduit data store nodes to have non-uniform sizes");

  // Input options
  arg_parser.add_option(
    LBANN_OPTION_DATA_STORE_SPILL,
    {"--data_store_spill"},
    "[DATASTORE] Base directory for conduit data store to spill data",
    "");
  arg_parser.add_option(
    LBANN_OPTION_DATA_STORE_TEST_CHECKPOINT,
    {"--data_store_test_checkpoint"},
    "[DATASTORE] Set directory for running checks on conduit data store "
    "checkpointing, used for testing purposes",
    "");
}

void construct_datareader_options()
{
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(
    LBANN_OPTION_CHECK_DATA,
    {"--check_data"},
    "[DATAREADER] Checks if the data file exists for image datareader");
  arg_parser.add_flag(
    LBANN_OPTION_KEEP_SAMPLE_ORDER,
    {"--keep_sample_order"},
    "[DATAREADER] Makes sure the order of samples in the list remains the same "
    "even with loading in an interleaving order by multiple trainer workers");
  arg_parser.add_flag(
    LBANN_OPTION_KEEP_PACKED_FIELDS,
    {"--keep_packed_fields"},
    "[DATAREADER] Prevents packed fields deletion in HDF5 data reader");
  arg_parser.add_flag(
    LBANN_OPTION_LOAD_FULL_SAMPLE_LIST_ONCE,
    {"--load_full_sample_list_once"},
    "[DATAREADER] Trainer master will load entire sample list into memory and "
    "then broadcast it to other workers within the trainer");
  arg_parser.add_flag(
    LBANN_OPTION_QUIET,
    {"--quiet"},
    "[DATAREADER] Silences metadata output from HDF5 datareader");
  arg_parser.add_flag(LBANN_OPTION_WRITE_SAMPLE_LABEL_LIST,
                      {"--write_sample_label_list"},
                      "[DATAREADER] When enabled, the sample labels from image "
                      "datareader are output to file in current directory");
  arg_parser.add_flag(LBANN_OPTION_WRITE_SAMPLE_LIST,
                      {"--write_sample_list"},
                      "[DATAREADER] Writes out the sample list into file in "
                      "current directory for image datareader");
  arg_parser.add_flag(
    LBANN_OPTION_Z_SCORE,
    {"--z_score"},
    "[DATAREADER] RAS lipid conduit data reader will normalize data with "
    "z-score, default normalizes with max-min");

  // Input options
  arg_parser.add_option(LBANN_OPTION_ABSOLUTE_SAMPLE_COUNT,
                        {"--absolute_sample_count"},
                        "[DATAREADER] Number of data samples to use",
                        -1);
  arg_parser.add_option(
    LBANN_OPTION_DATA_FILEDIR,
    {"--data_filedir"},
    "[DATAREADER] Sets the file directory for train and test data",
    "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILEDIR_TEST,
                        {"--data_filedir_test"},
                        "[DATAREADER] Sets the file directory for test data",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILEDIR_TRAIN,
                        {"--data_filedir_train"},
                        "[DATAREADER] Sets the file directory for train data",
                        "");
  arg_parser.add_option(
    LBANN_OPTION_DATA_FILEDIR_VALIDATE,
    {"--data_filedir_validate"},
    "[DATAREADER] Sets the file directory for validation data",
    "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILENAME_TEST,
                        {"--data_filename_test"},
                        "[DATAREADER] Sets the filename for test data",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILENAME_TRAIN,
                        {"--data_filename_train"},
                        "[DATAREADER] Sets the filename for train data",
                        "");
  arg_parser.add_option(LBANN_OPTION_DATA_FILENAME_VALIDATE,
                        {"--data_filename_validate"},
                        "[DATAREADER] Sets the filename for validation data",
                        "");
  arg_parser.add_option(
    LBANN_OPTION_DATA_READER_FRACTION,
    {"--data_reader_fraction"},
    "[DATAREADER] Sets the fraction of total samples to use",
    (float)-1);
  arg_parser.add_option(
    LBANN_OPTION_LABEL_FILENAME_TEST,
    {"--label_filename_test"},
    "[DATAREADER] Sets the filename for testing data labels",
    "");
  arg_parser.add_option(
    LBANN_OPTION_LABEL_FILENAME_TRAIN,
    {"--label_filename_train"},
    "[DATAREADER] Sets the filename for training data labels",
    "");
  arg_parser.add_option(
    LBANN_OPTION_LABEL_FILENAME_VALIDATE,
    {"--label_filename_validate"},
    "[DATAREADER] Sets the filename for validation data labels",
    "");
  arg_parser.add_option(LBANN_OPTION_NORMALIZATION,
                        {"--normalization"},
                        "[DATAREADER] Sets the filename for normalization data "
                        "with RAS lipid datareader",
                        "");
  arg_parser.add_option(LBANN_OPTION_PILOT2_READ_FILE_SIZES,
                        {"--pilot2_read_file_sizes"},
                        "[DATAREADER] Sets the filename for loading number of "
                        "samples per file for RAS lipid datatreader",
                        "");
  arg_parser.add_option(LBANN_OPTION_PILOT2_SAVE_FILE_SIZES,
                        {"--pilot2_save_file_sizes"},
                        "[DATAREADER] Sets the filename for saving computed "
                        "number of samples per file for RAS lipid datatreader",
                        "");
  arg_parser.add_option(
    LBANN_OPTION_SAMPLE_LIST_TEST,
    {"--sample_list_test"},
    "[DATAREADER] Sets the datareader sample list for test data",
    "");
  arg_parser.add_option(
    LBANN_OPTION_SAMPLE_LIST_TRAIN,
    {"--sample_list_train"},
    "[DATAREADER] Sets the datareader sample list for training data",
    "");
  arg_parser.add_option(
    LBANN_OPTION_SAMPLE_LIST_VALIDATE,
    {"--sample_list_validate"},
    "[DATAREADER] Sets the datareader sample list for validation data",
    "");
  arg_parser.add_option(LBANN_OPTION_SEQUENCE_LENGTH,
                        {"--sequence_length", "--seq_len"},
                        "[DATAREADER] Sets the sequence length for RAS lipid "
                        "and SMILES datareaders",
                        -1);
  arg_parser.add_option(LBANN_OPTION_SMILES_BUFFER_SIZE,
                        {"--smiles_buffer_size"},
                        utils::ENV("LBANN_SMILES_BUFFER_SIZE"),
                        "[DATAREADER] Size of the read buffer for the SMILES "
                        "data reader.",
                        16 * 1024 * 1024UL);
  arg_parser.add_option(LBANN_OPTION_VOCAB,
                        {"--vocab"},
                        "[DATAREADER] Sets the filename containing the "
                        "vocabulary for SMILES datareader",
                        "");
}

void construct_jag_options()
{
  auto& arg_parser = global_argument_parser();

  // Bool flags
  arg_parser.add_flag(LBANN_OPTION_JAG, {"--jag"}, "[JAG] Runs the JAG test");

  // Input options
  arg_parser.add_option(
    LBANN_OPTION_BASE_DIR,
    {"--base_dir"},
    "[JAG] Sets the file path to directory containing conduit files",
    "");
  arg_parser.add_option(LBANN_OPTION_FILELIST,
                        {"--filelist"},
                        "[JAG] List of Conduit filenames",
                        "");
  arg_parser.add_option(LBANN_OPTION_FILENAME,
                        {"--filename"},
                        "[JAG] Sets HDF5 file for JAG Conduit HDF5 test",
                        "");
  arg_parser.add_option(
    LBANN_OPTION_FORMAT,
    {"--format"},
    "[JAG] Sets format for test_speed_hydra <hdf5|conduit_bin>",
    "");
  arg_parser.add_option(
    LBANN_OPTION_INDEX_FN,
    {"--index_fn"},
    "[JAG] Sets filename for output file from build_index executable",
    "");
  arg_parser.add_option(LBANN_OPTION_MAPPING_FN,
                        {"--mapping_fn"},
                        "[JAG] Sets filename for mapping values",
                        "");
  arg_parser.add_option(
    LBANN_OPTION_NUM_LISTS,
    {"--num_lists"},
    "[JAG] Number of sets for randomly selected samples to be partitioned",
    -1);
  arg_parser.add_option(LBANN_OPTION_NUM_SAMPLES,
                        {"--num_samples"},
                        "[JAG] Number of random samples to be extracted",
                        -1);
  arg_parser.add_option(LBANN_OPTION_NUM_SAMPLES_PER_FILE,
                        {"--num_samples_per_file"},
                        "[JAG] Number of samples per output file",
                        1000);
  arg_parser.add_option(LBANN_OPTION_NUM_SAMPLES_PER_LIST,
                        {"--num_samples_per_list"},
                        "[JAG] Sets the number of samples per list",
                        -1);
  arg_parser.add_option(LBANN_OPTION_NUM_SUBDIRS,
                        {"--num_subdirs"},
                        "[JAG] Sets the number of output directories",
                        -1);
  arg_parser.add_option(
    LBANN_OPTION_OUTPUT_BASE_DIR,
    {"--output_base_dir"},
    "[JAG] Sets path for output directory, will be created if it doesn't exist",
    "");
  arg_parser.add_option(LBANN_OPTION_OUTPUT_BASE_FN,
                        {"--output_base_fn"},
                        "[JAG] Sets output filename for sample selection",
                        "");
  arg_parser.add_option(
    LBANN_OPTION_OUTPUT_DIR,
    {"--output_dir"},
    "[JAG] Sets the output directory for various JAG util executables",
    "");
  arg_parser.add_option(LBANN_OPTION_OUTPUT_FN,
                        {"--output_fn"},
                        "[JAG] Sets output filename for build_index executable",
                        "");
  arg_parser.add_option(LBANN_OPTION_SAMPLES_PER_FILE,
                        {"--samples_per_file"},
                        "[JAG] Sets number of samples per Conduit output file",
                        -1);
}

void construct_all_options()
{
  construct_std_options();
  construct_datastore_options();
  construct_datareader_options();
  // construct_jag_options();
}

} // namespace lbann
