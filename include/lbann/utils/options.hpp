#ifndef LBANN_UTILS_OPTIONS_HPP_INCLUDED
#define LBANN_UTILS_OPTIONS_HPP_INCLUDED

#include "lbann/utils/argument_parser.hpp"

#include <iostream>
#include <map>
#include <vector>
#include <string>

namespace lbann {

/****** std options ******/
// Bool flags
#define DISABLE_BACKGROUND_IO_ACTIVITY "disable_background_io_activity"
#define DISABLE_CUDA "disable_cuda"
#define FN "fn"
#define LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE "load_model_weights_dir_is_complete"
#define LTFB_ALLOW_GLOBAL_STATISTICS "LTFB Allow global statistics"
#define LTFB_VERBOSE "ltfb_verbose"
#define NO_IM_COMM "no_im_comm"
#define PRELOAD_DATA_STORE "preload_data_store"
#define PRINT_AFFINITY "print_affinity"
#define SERIALIZE_IO "serialize_io"
#define ST_FULL_TRACE "st_full_trace"
#define ST_ON "st_on"
#define USE_CUBLAS_TENSOR_OPS "use_cublas_tensor_ops"
#define USE_CUDNN_TENSOR_OPS "use_cudnn_tensor_ops"
#define USE_DATA_STORE "use_data_store"
#define USE_LTFB "ltfb"
#define VERBOSE "verbose"
#define WRITE_SAMPLE_LIST "write_sample_list"

// Input options
#define CKPT_DIR "ckpt_dir"
#define HYDROGEN_BLOCK_SIZE "hydrogen_block_size"
#define LOAD_MODEL_WEIGHTS_DIR "load_model_weights_dir"
#define MAX_RNG_SEEDS_DISPLAY "RNG seeds per trainer to display"
#define METADATA "metadata"
#define MINI_BATCH_SIZE "mini_batch_size"
#define MODEL "model"
#define NUM_EPOCHS "num_epochs"
#define NUM_IO_THREADS "Num. IO threads"
#define NUM_PARALLEL_READERS "num_parallel_readers"
#define NUM_TEST_SAMPLES "Num test samples"
#define NUM_TRAIN_SAMPLES "Num train samples"
#define NUM_VALIDATE_SAMPLES "Num validate samples"
#define OPTIMIZER "optimizer"
#define PROCS_PER_TRAINER "Processes per trainer"
#define PROTOTEXT "prototext"
#define RANDOM_SEED "random_seed"
#define READER "reader"
#define RESTART_DIR "restart_dir"
#define TRAINER_CREATE_TWO_MODELS "Create two models in Sub-grid parallelism"
#define TRAINER_GRID_HEIGHT "Height of 2D process grid for each trainer"
#define TRAINER_PRIMARY_GRID_SIZE "Primary Grid Size per trainer"

/****** datastore options ******/
// Bool flags
#define DATA_STORE_CACHE "data_store_cache"
#define DATA_STORE_DEBUG "data_store_debug"
#define DATA_STORE_FAIL "data_store_fail"
#define DATA_STORE_MIN_MAX_TIMING "data_store_min_max_timing"
#define DATA_STORE_NO_THREAD "data_store_no_thread"
#define DATA_STORE_PROFILE "data_store_profile"
#define DATA_STORE_SPILL "data_store_spill"
#define DATA_STORE_TEST_CACHE "data_store_test_cache"
#define DATA_STORE_TEST_CHECKPOINT "data_store_test_checkpoint"

/****** datareader options ******/
// Bool flags
#define ALL_GATHER_OLD "all_gather_old"
#define CHECK_DATA "check_data"
#define CREATE_TARBALL "create_tarball"
#define DISABLE_SIGNAL_HANDLER "disable_signal_handler"
#define DEBUG_CONCATENATE "debug_concatenate"
#define EXIT_AFTER_SETUP "exit_after_setup"
#define GENERATE_MULTI_PROTO "generate_multi_proto"
#define KEEP_SAMPLE_ORDER "keep_sample_order"
#define KEEP_PACKED_FIELDS "keep_packed_fields"
#define LOAD_FULL_SAMPLE_LIST_ONCE "load_full_sample_list_once"
#define MAKE_TEST_FAIL "make_test_fail"
#define NODE_SIZES_VARY "node_sizes_vary"
#define QUIET "quiet"
#define STACK_TRACE_TO_FILE "stack_trace_to_file"
#define TEST_ENCODE "test_encode"
#define WRITE_SAMPLE_LABEL_LIST "write_sample_label_list"
#define Z_SCORE "z_score"

// Input options
#define ABSOLUTE_SAMPLE_COUNT "absolute_sample_count"
#define DATA_FILEDIR "data_filedir"
#define DATA_FILEDIR_TEST "data_filedir_test"
#define DATA_FILEDIR_TRAIN "data_filedir_train"
#define DATA_FILEDIR_VALIDATE "data_filedir_validate"
#define DATA_FILENAME_TEST "data_filename_test"
#define DATA_FILENAME_TRAIN "data_filename_train"
#define DATA_FILENAME_VALIDATE "data_filename_validate"
#define DATA_READER_PERCENT "data_reader_percent"
#define DELIMITER "delimiter"
#define IMAGE_SIZES_FILENAME "image_sizes_filename"
#define LABEL_FILENAME_TEST "label_filename_test"
#define LABEL_FILENAME_TRAIN "label_filename_train"
#define LABEL_FILENAME_VALIDATE "label_filename_validate"
#define NORMALIZATION "normalization"
#define N_LINES "n_lines"
#define PAD_INDEX "pad_index"
#define PILOT2_READ_FILE_SIZES "pilot2_read_file_sizes"
#define PILOT2_SAVE_FILE_SIZES "pilot2_save_file_sizes"
#define SAMPLE_LIST_TEST "sample_list_test"
#define SAMPLE_LIST_TRAIN "sample_list_train"
#define SAMPLE_LIST_VALIDATE "sample_list_validate"
#define SEQUENCE_LENGTH "sequence_length"
#define SMILES_BUFFER_SIZE "smiles_buffer_size"
#define TEST_TARBALL "test_tarball"
#define VOCAB "vocab"

/****** jag options ******/
// Bool flags
#define JAG "jag"
#define JAG_PARTITIONED "jag_partitioned"

// Input options
#define BASE_DIR "base_dir"
#define FILELIST "filelist"
#define FILENAME "filename"
#define FORMAT "format"
#define INDEX_FN "index_fn"
#define MAPPING_FN "mapping_fn"
#define NUM_LISTS "num_lists"
#define NUM_SAMPLES "num_samples"
#define NUM_SAMPLES_PER_FILE "num_samples_per_file"
#define NUM_SAMPLES_PER_LIST "num_samples_per_list"
#define NUM_SUBDIRS "num_subdirs"
#define OUTPUT_BASE_DIR "output_base_dir"
#define OUTPUT_BASE_FN "output_base_fn"
#define OUTPUT_DIR "output_dir"
#define OUTPUT_FN "output_fn"
#define SAMPLES_PER_FILE "samples_per_file"

void construct_std_options();
void construct_datastore_options();
void construct_datareader_options();
void construct_jag_options();
void construct_all_options();

} // namespace lbann

#endif // LBANN_UTILS_OPTIONS_HPP_INCLUDED
