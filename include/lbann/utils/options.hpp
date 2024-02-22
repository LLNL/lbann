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
#ifndef LBANN_UTILS_OPTIONS_HPP_INCLUDED
#define LBANN_UTILS_OPTIONS_HPP_INCLUDED

#include "lbann/utils/argument_parser.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace lbann {

/****** std options ******/
// Bool flags
#define LBANN_OPTION_DISABLE_BACKGROUND_IO_ACTIVITY                            \
  "disable_background_io_activity"
#define LBANN_OPTION_DISABLE_CUDA "disable_cuda"
#define LBANN_OPTION_DISABLE_DISTCONV "disable_distconv"
#define LBANN_OPTION_DISABLE_SIGNAL_HANDLER "disable_signal_handler"
#define LBANN_OPTION_EXIT_AFTER_SETUP "exit_after_setup"
#define LBANN_OPTION_GENERATE_MULTI_PROTO "generate_multi_proto"
#define LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE                        \
  "load_model_weights_dir_is_complete"
// Deprecated -- "LTFB Callback"
#define LBANN_OPTION_LTFB_ALLOW_GLOBAL_STATISTICS "LTFB Allow global statistics"
// Deprecated -- "LTFB Callback"
#define LBANN_OPTION_LTFB_VERBOSE "ltfb_verbose"
#define LBANN_OPTION_MULTITRAINER_VERBOSE "multitrainer_verbose"
#define LBANN_OPTION_ALLOW_MULTITRAINER_GLOBAL_STATISTICS                      \
  "Allow multitrainer global statistics"
#define LBANN_OPTION_PRELOAD_DATA_STORE "preload_data_store"
#define LBANN_OPTION_PRINT_AFFINITY "print_affinity"
#define LBANN_OPTION_SERIALIZE_IO "serialize_io"
#define LBANN_OPTION_STACK_TRACE_TO_FILE "stack_trace_to_file"
#define LBANN_OPTION_DISABLE_CUDNN_TENSOR_OPS "disable_cudnn_tensor_ops"
#define LBANN_OPTION_USE_DATA_STORE "use_data_store"
#define LBANN_OPTION_VERBOSE "verbose"
#define LBANN_OPTION_USE_GPU_DIRECT_MEMORY_IN_FORWARD_PROP                     \
  "Use direct (i.e., not pooled) memory mode for GPU buffers in forward prop"
#define LBANN_OPTION_INIT_SHMEM "Initialize SHMEM when initializing LBANN"
#define LBANN_OPTION_INIT_NVSHMEM "Initialize NVSHMEM when initializing LBANN"
#define LBANN_OPTION_NO_INPLACE "no_inplace"
#define LBANN_OPTION_NO_BACKPROP_DISABLE "no_backprop_disable"

#define LBANN_OPTION_OMP_NUM_THREADS "Num. OMP threads"

// Input options
#define LBANN_OPTION_CKPT_DIR "ckpt_dir"
#define LBANN_OPTION_HYDROGEN_BLOCK_SIZE "hydrogen_block_size"
#define LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR "load_model_weights_dir"
#define LBANN_OPTION_MAX_RNG_SEEDS_DISPLAY "RNG seeds per trainer to display"
#define LBANN_OPTION_METADATA "metadata"
#define LBANN_OPTION_MINI_BATCH_SIZE "mini_batch_size"
#define LBANN_OPTION_MODEL "model"
#define LBANN_OPTION_NUM_EPOCHS "num_epochs"
#define LBANN_OPTION_NUM_IO_THREADS "Num. IO threads"
#define LBANN_OPTION_OPTIMIZER "optimizer"
#define LBANN_OPTION_PROCS_PER_TRAINER "Processes per trainer"
#define LBANN_OPTION_PROTOTEXT "prototext"
#define LBANN_OPTION_RANDOM_SEED "random_seed"
#define LBANN_OPTION_READER "reader"
#define LBANN_OPTION_RESTART_DIR "restart_dir"
#define LBANN_OPTION_TRAINER_CREATE_TWO_MODELS                                 \
  "Create two models in Sub-grid parallelism"
#define LBANN_OPTION_TRAINER_GRID_HEIGHT                                       \
  "Height of 2D process grid for each trainer"
#define LBANN_OPTION_TRAINER_PRIMARY_GRID_SIZE "Primary Grid Size per trainer"
#define LBANN_OPTION_TRAINER_ENABLE_SUBGRID_ASYNC_COMM                         \
  "Enable async communication in Sub-grid parallelism"
#define LBANN_OPTION_TRAINER_ENABLE_TOPO_AWARE_SUBGRID                         \
  "Enable topology aware process placement in Sub-grid parallelism"
#define LBANN_OPTION_NUM_SUBGRIDS_BLOCK_ORDER                                  \
  "Divide each trainer into equally-sized sub-grids with blocked ordering"
#ifdef LBANN_HAS_CALIPER
#define LBANN_OPTION_USE_CALIPER "use caliper"
#define LBANN_OPTION_CALIPER_CONFIG "caliper_config"
#endif

/****** datastore options ******/
// Bool flags
#define LBANN_OPTION_DATA_STORE_CACHE "data_store_cache"
#define LBANN_OPTION_DATA_STORE_DEBUG "data_store_debug"
#define LBANN_OPTION_DATA_STORE_FAIL "data_store_fail"
#define LBANN_OPTION_DATA_STORE_MIN_MAX_TIMING "data_store_min_max_timing"
#define LBANN_OPTION_DATA_STORE_NO_THREAD "data_store_no_thread"
#define LBANN_OPTION_DATA_STORE_PROFILE "data_store_profile"
#define LBANN_OPTION_DATA_STORE_TEST_CACHE "data_store_test_cache"
#define LBANN_OPTION_NODE_SIZES_VARY "node_sizes_vary"

// Input options
#define LBANN_OPTION_DATA_STORE_SPILL "data_store_spill"
#define LBANN_OPTION_DATA_STORE_TEST_CHECKPOINT "data_store_test_checkpoint"

/****** datareader options ******/
// Bool flags
#define LBANN_OPTION_CHECK_DATA "check_data"
#define LBANN_OPTION_KEEP_SAMPLE_ORDER "keep_sample_order"
#define LBANN_OPTION_KEEP_PACKED_FIELDS "keep_packed_fields"
#define LBANN_OPTION_LOAD_FULL_SAMPLE_LIST_ONCE "load_full_sample_list_once"
#define LBANN_OPTION_QUIET "quiet"
#define LBANN_OPTION_WRITE_SAMPLE_LABEL_LIST "write_sample_label_list"
#define LBANN_OPTION_WRITE_SAMPLE_LIST "write_sample_list"
#define LBANN_OPTION_Z_SCORE "z_score"

// Input options
#define LBANN_OPTION_ABSOLUTE_SAMPLE_COUNT "absolute_sample_count"
#define LBANN_OPTION_DATA_FILEDIR "data_filedir"
#define LBANN_OPTION_DATA_FILEDIR_TEST "data_filedir_test"
#define LBANN_OPTION_DATA_FILEDIR_TRAIN "data_filedir_train"
#define LBANN_OPTION_DATA_FILEDIR_VALIDATE "data_filedir_validate"
#define LBANN_OPTION_DATA_FILENAME_TEST "data_filename_test"
#define LBANN_OPTION_DATA_FILENAME_TRAIN "data_filename_train"
#define LBANN_OPTION_DATA_FILENAME_VALIDATE "data_filename_validate"
#define LBANN_OPTION_DATA_READER_FRACTION "data_reader_fraction"
#define LBANN_OPTION_LABEL_FILENAME_TEST "label_filename_test"
#define LBANN_OPTION_LABEL_FILENAME_TRAIN "label_filename_train"
#define LBANN_OPTION_LABEL_FILENAME_VALIDATE "label_filename_validate"
#define LBANN_OPTION_NORMALIZATION "normalization"
#define LBANN_OPTION_PILOT2_READ_FILE_SIZES "pilot2_read_file_sizes"
#define LBANN_OPTION_PILOT2_SAVE_FILE_SIZES "pilot2_save_file_sizes"
#define LBANN_OPTION_SAMPLE_LIST_TEST "sample_list_test"
#define LBANN_OPTION_SAMPLE_LIST_TRAIN "sample_list_train"
#define LBANN_OPTION_SAMPLE_LIST_VALIDATE "sample_list_validate"
#define LBANN_OPTION_SEQUENCE_LENGTH "sequence_length"
#define LBANN_OPTION_SMILES_BUFFER_SIZE "smiles_buffer_size"
#define LBANN_OPTION_VOCAB "vocab"

/****** jag options ******/
// Bool flags
#define LBANN_OPTION_JAG "jag"

// Input options
#define LBANN_OPTION_BASE_DIR "base_dir"
#define LBANN_OPTION_FILELIST "filelist"
#define LBANN_OPTION_FILENAME "filename"
#define LBANN_OPTION_FORMAT "format"
#define LBANN_OPTION_INDEX_FN "index_fn"
#define LBANN_OPTION_MAPPING_FN "mapping_fn"
#define LBANN_OPTION_NUM_LISTS "num_lists"
#define LBANN_OPTION_NUM_SAMPLES "num_samples"
#define LBANN_OPTION_NUM_SAMPLES_PER_FILE "num_samples_per_file"
#define LBANN_OPTION_NUM_SAMPLES_PER_LIST "num_samples_per_list"
#define LBANN_OPTION_NUM_SUBDIRS "num_subdirs"
#define LBANN_OPTION_OUTPUT_BASE_DIR "output_base_dir"
#define LBANN_OPTION_OUTPUT_BASE_FN "output_base_fn"
#define LBANN_OPTION_OUTPUT_DIR "output_dir"
#define LBANN_OPTION_OUTPUT_FN "output_fn"
#define LBANN_OPTION_SAMPLES_PER_FILE "samples_per_file"

void construct_std_options();
void construct_datastore_options();
void construct_datareader_options();
void construct_jag_options();
void construct_all_options();

} // namespace lbann

#endif // LBANN_UTILS_OPTIONS_HPP_INCLUDED
