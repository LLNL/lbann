#ifndef LBANN_UTILS_OPTIONS_HPP_INCLUDED
#define LBANN_UTILS_OPTIONS_HPP_INCLUDED

#include "lbann/utils/argument_parser.hpp"

#include <iostream>
#include <map>
#include <vector>
#include <string>

namespace lbann {

#define MAX_RNG_SEEDS_DISPLAY "RNG seeds per trainer to display"
#define NUM_IO_THREADS "Num. IO threads"
#define NUM_TRAIN_SAMPLES "Num train samples"
#define NUM_VALIDATE_SAMPLES "Num validate samples"
#define NUM_TEST_SAMPLES "Num test samples"
#define ALLOW_GLOBAL_STATISTICS "LTFB Allow global statistics"
#define PROCS_PER_TRAINER "Processes per trainer"
#define TRAINER_GRID_HEIGHT "Height of 2D process grid for each trainer"

void construct_callback_options();
void construct_jag_options();
void construct_datareader_options();
void construct_std_options();
void construct_all_options();

} // namespace lbann

#endif // LBANN_UTILS_OPTIONS_HPP_INCLUDED
