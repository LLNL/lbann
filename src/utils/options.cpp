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
  arg_parser.add_required_argument<std::string>
                                  ("prototext",
                                   "Prototext file containing experiment");
  arg_parser.add_flag("help",
                      {"--help", "-h"},
                      "Prints the help message");
  arg_parser.add_flag("verbose",
                      {"--verbose"},
                      "Turns on verbose mode");
  arg_parser.add_option(MAX_RNG_SEEDS_DISPLAY,
                        {"--rng_seeds_per_trainer_to_display"},
                        utils::ENV("LBANN_RNG_SEEDS_PER_TRAINER_TO_DISPLAY"),
                        "Limit how many random seeds LBANN should display "
                        "from each trainer",
                        2);
  arg_parser.add_option(NUM_IO_THREADS,
                        {"--num_io_threads"},
                        utils::ENV("LBANN_NUM_IO_THREADS"),
                        "Number of threads available to both I/O and "
                        "initial data transformations for each rank.",
                        64);
  arg_parser.add_option(NUM_TRAIN_SAMPLES,
                        {"--num_train_samples"},
                        utils::ENV("LBANN_NUM_TRAIN_SAMPLES"),
                        "Set the number of training samples to ingest.",
                        0);
  arg_parser.add_option(NUM_VALIDATE_SAMPLES,
                        {"--num_validate_samples"},
                        utils::ENV("LBANN_NUM_VALIDATE_SAMPLES"),
                        "Set the number of validate samples to ingest.",
                        0);
  arg_parser.add_option(NUM_TEST_SAMPLES,
                        {"--num_test_samples"},
                        utils::ENV("LBANN_NUM_TEST_SAMPLES"),
                        "Set the number of testing samples to ingest.",
                        0);
  arg_parser.add_flag(ALLOW_GLOBAL_STATISTICS,
                      {"--ltfb_allow_global_statistics"},
                      utils::ENV("LBANN_LTFB_ALLOW_GLOBAL_STATISTICS"),
                      "Allow the print_statistics callback to report "
                      "global (inter-trainer) summary statistics.");
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
                        0);
  arg_parser.add_option(TRAINER_GRID_HEIGHT,
                        {"--trainer_grid_height"},
                        utils::ENV("LBANN_TRAINER_GRID_HEIGHT"),
                        "Height of 2D process grid for each trainer. "
                        "Default grid is approximately square.",
                        -1);
  arg_parser.add_option("smiles_buffer_size",
                        {"--smiles_buffer_size"},
                        utils::ENV("LBANN_SMILES_BUFFER_SIZE"),
                        "Size of the read buffer for the SMILES "
                        "data reader.",
                        16*1024*1024UL);
  arg_parser.add_option("mini_batch_size",
                        {"--mini_batch_size"},
                        "Size of mini batches",
                        0);
  arg_parser.add_option("num_epochs",
                        {"--num_epochs"},
                        "Number of epochs to train model",
                        0);
  arg_parser.add_option("hydrogen_block_size",
                        {"--hydrogen_block_size"},
                        "Block size for Hydrogen",
                        0);
  arg_parser.add_option("num_parallel_readers",
                        {"--num_parallel_readers"},
                        "The number of parallel data readers",
                        1);
  arg_parser.add_option("random_seed",
                        {"--random_seed"},
                        "Value to seed RNG",
                        -1);
  arg_parser.add_option("objective_function",
                        {"--objective_function"},
                        "must be: categorical_cross_entropy or mean_squared_error",
                        "categorical_cross_entropy");
  arg_parser.add_option("data_layout",
                        {"--data_layout"},
                        "must be: data_parallel or model_parallel\n"
                        "note: this will be applied to all layers, metrics (and others)\n"
                        "that take DATA_PARALLEL or MODEL_PARALLEL as a template parameter",
                        "data_parallel");
  arg_parser.add_option("checkpoint_dir",
                        {"--checkpoint_dir"},
                        "Save to or restart from a specific checkpoint directory.\n"
                        "Additionally, sets the output directory for dumping weights.\n"
                        "Modifies callbacks: checkpoint, save_model, dump_weights\n",
                        "");
  arg_parser.add_option("restart_dir",
                        {"--restart_dir"},
                        "Restart from a checkpoint found in the given directory.\n"
                        "If the directory doesn't exist or doesn't contain a checkpoint,\n"
                        "an error will be thrown.\n",
                        "");
  arg_parser.add_option("load_model_weights_dir",
                        {"--load_model_weights_dir"},
                        "Load model wieghts found in the given directory.\n"
                        "If the directory doesn't exist, doesn't contain valid weights,\n"
                        "or doesn't contain a checkpoint,\n"
                        "an error will be thrown.\n",
                        "");
  arg_parser.add_flag("serialize_io",
                      {"--serialize_io"},
                      "force data readers to use a single threader for I/O");
  arg_parser.add_flag("disable_background_io_activity",
                      {"--disable_background_io_activity"},
                      "prevent the input layers from fetching data in the background");
  arg_parser.add_flag("disable_cuda",
                      {"--disable_cuda"},
                      "has no effect unless LBANN was compiled with LBANN_HAS_CUDNN");
  arg_parser.add_flag("print_affinity",
                      {"--print_affinity"},
                      "display information on how OpenMP threads are provisioned");
  arg_parser.add_flag("use_data_store",
                      {"--use_data_store"},
                      "Enables the data store in-memory structure");
  arg_parser.add_flag("preload_data_store",
                      {"--preload_data_store"},
                      "Preloads the data store in-memory structure druing data reader load time");
  arg_parser.add_flag("super_node",
                      {"--super_node"},
                      "Enables the data store in-memory structure to use the supernode exchange structure");
  arg_parser.add_flag("write_sample_list",
                      {"--write_sample_list"},
                      "Writes out the sample list that was loaded into the current directory");
  arg_parser.add_flag("ltfb_verbose",
                      {"--ltfb_verbose"},
                      "Increases number of per-trainer messages that are reported");
  arg_parser.add_flag("load_model_weights_dir_is_complete",
                      {"--load_model_weights_dir_is_complete"},
                      "Use load_model_weights_dir as given, ignoring checkpoint hierarchy");
}

void construct_datareader_options() {
  auto& arg_parser = global_argument_parser();
  arg_parser.add_option("data_filedir",
                        {"--data_filedir"},
                        "Sets the file direcotry for train and test data",
                        "");
  arg_parser.add_option("data_filedir_train",
                        {"--data_filedir_train"},
                        "TODO",
                        "");
  arg_parser.add_option("data_filename_train",
                        {"--data_filename_train"},
                        "TODO",
                        "");
  arg_parser.add_option("sample_list_train",
                        {"--sample_list_train"},
                        "TODO",
                        "");
  arg_parser.add_option("label_filename_train",
                        {"--label_filename_train"},
                        "TODO",
                        "");
  arg_parser.add_option("data_reader_percent",
                        {"--data_reader_percent"},
                        "TODO",
                        0);
  arg_parser.add_flag("share_testing_data_readers",
                      {"--share_testing_data_readers"},
                      "TODO");
  arg_parser.add_flag("create_tarball",
                      {"--create_tarball"},
                      "TODO");
  arg_parser.add_option("test_tarball",
                        {"--test_tarball"},
                        "TODO",
                        0);
  arg_parser.add_flag("all_gather_old",
                      {"--all_gather_old"},
                      "TODO");
  arg_parser.add_flag("disable_signal_handler",
                      {"--disable_signal_handler"},
                      "TODO");
  arg_parser.add_flag("stack_trace_to_file",
                      {"--stack_trace_to_file"},
                      "TODO");
  arg_parser.add_flag("generate_multi_proto",
                      {"--generate_multi_proto"},
                      "TODO");
  arg_parser.add_flag("exit_after_setup",
                      {"--exit_after_setup"},
                      "TODO");
  arg_parser.add_flag("node_sizes_vary",
                      {"--node_size_vary"},
                      "TODO");
}

void construct_jag_options() {
  auto& arg_parser = global_argument_parser();
  arg_parser.add_flag("jag",
                      {"--jag"},
                      "TODO");
  arg_parser.add_option("filelist",
                        {"--filelist"},
                        "TODO",
                        "");
  arg_parser.add_option("filename",
                        {"--filename"},
                        "TODO",
                        "");
  arg_parser.add_option("output_fn",
                        {"--output_fn"},
                        "TODO",
                        "");
  arg_parser.add_option("index_fn",
                        {"--index_fn"},
                        "TODO",
                        "");
  arg_parser.add_option("mapping_fn",
                        {"--mapping_fn"},
                        "TODO",
                        "");
  arg_parser.add_option("base_dir",
                        {"--base_dir"},
                        "TODO",
                        "");
  arg_parser.add_option("output_dir",
                        {"--output_dir"},
                        "TODO",
                        "");
  arg_parser.add_option("output_base_dir",
                        {"--output_base_dir"},
                        "TODO",
                        "");
  arg_parser.add_option("output_base_fn",
                        {"--output_base_fn"},
                        "TODO",
                        "");
  arg_parser.add_option("num_lists",
                        {"--num_lists"},
                        "TODO",
                        -1);
  arg_parser.add_option("num_subdirs",
                        {"--num_subdirs"},
                        "TODO",
                        -1);
  arg_parser.add_option("format",
                        {"--format"},
                        "TODO",
                        "");
  arg_parser.add_option("num_samples_per_file",
                        {"--num_samples_per_file"},
                        "TODO",
                        -1);
  arg_parser.add_option("num_samples_per_list",
                        {"--num_samples_per_list"},
                        "TODO",
                        -1);
  arg_parser.add_option("samples_per_file",
                        {"--samples_per_file"},
                        "TODO",
                        -1);
  arg_parser.add_option("num_samples",
                        {"--num_samples"},
                        "TODO",
                        -1);
  arg_parser.add_option("rand_seed",
                        {"--rand_seed"},
                        "TODO",
                        -1);
  arg_parser.add_option("random_seed",
                        {"--random_seed"},
                        "TODO",
                        -1);
}

void construct_callback_options() {
  auto& arg_parser = global_argument_parser();
  arg_parser.add_option("image_dir",
                        {"--image_dir"},
                        "if the model has callback_save_images, this determines where the"
                        "images are saved",
                        "");
  arg_parser.add_flag("no_im_comm",
                      {"--no_im_comm"},
                      "removed ImComm callback, if present; this is intended for"
                      "running alexnet with a single model, but may be useful elsewhere");
}

void construct_all_options() {
  construct_callback_options();
  construct_datareader_options();
  construct_jag_options();
  construct_std_options();
}

} // namespace lbann
