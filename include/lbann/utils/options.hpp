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
#define HELP "help"
#define LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE "load_model_weights_dir_is_complete"
#define LTFB "ltfb"
#define LTFB_VERBOSE "ltfb_verbose"
#define PRELOAD_DATA_STORE "preload_data_store"
#define PRINT_AFFINITY "print_affinity"
#define SERIALIZE_IO "serialize_io"
#define ST_FULL_TRACE "st_full_trace"
#define ST_ON "st_on"
#define USE_DATA_STORE "use_data_store"
#define VERBOSE "verbose"
#define WRITE_SAMPLE_LIST "write_sample_list"

// Input options
#define CHECKPOINT_DIR "checkpoint_dir"
#define DATA_LAYOUT "data_layout"
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
#define OBJECTIVE_FUNCTION "objective_function"
#define OPTIMIZER "optimizer"
#define PROCS_PER_TRAINER "Processes per trainer"
#define PROTOTEXT "prototext"
#define RANDOM_SEED "random_seed"
#define READER "reader"
#define RESTART_DIR "restart_dir"
#define SMILES_BUFFER_SIZE "smiles_buffer_size"
#define TRAINER_GRID_HEIGHT "Height of 2D process grid for each trainer"

// Unused options
#define ALLOW_GLOBAL_STATISTICS "LTFB Allow global statistics"
#define SUPER_NODE "super_node"

void construct_callback_options();
void construct_jag_options();
void construct_datareader_options();
void construct_std_options();
void construct_all_options();

} // namespace lbann

#endif // LBANN_UTILS_OPTIONS_HPP_INCLUDED
