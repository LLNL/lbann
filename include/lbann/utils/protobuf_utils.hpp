////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UTILS_PROTOBUF_UTILS_HPP_INCLUDED
#define LBANN_UTILS_PROTOBUF_UTILS_HPP_INCLUDED

#include <memory>
#include <string>
#include <vector>

// Forward-declare protobuf class
namespace lbann_data {
class LbannPB;
}

namespace lbann {

/** @file protobuf_utils.hpp
 *  @brief static methods for parsing command line for prototext
 *         filenames, reading in prototext files, etc.
 */

struct prototext_fn_triple
{
  std::string model;
  std::string reader;
  std::string data_set_metadata;
  std::string optimizer;
};

namespace protobuf_utils {
/** @brief convience wrapper for other parsing, loading, and verifying
 * prototext.
 *
 *  Calls parse_prototext_filenames_from_command_line(),
 *  then load_prototext(), then verify_prototext(). This is the only function
 *  that needs to be called from, e.g, model_zoo/lbann.cpp; the three called
 *  functions are made public for testing.
 */
std::vector<std::unique_ptr<lbann_data::LbannPB>>
load_prototext(const bool master, const int trainer_rank = 0);

/** @brief Parses the command line for special prototext flags
 *
 *  This looks for `--model=<string>`, `--reader=<string>`, and
 *  `--optimizer=<string>` as well as their multi-value counterparts:
 *  `--model={<string_1>,<string_2>,...}`,
 *  `--reader={<string_1>,<string_2>,...}`, and
 *  `--optimizer={<string_1>,<string_2>,...}`. If the multi-model
 *  option is given, the reader and optimzier can either be single, or
 *  contain the same number of filenames as does the `--model={...}`
 *  specification.
 */
std::vector<prototext_fn_triple>
parse_prototext_filenames_from_command_line(const bool master,
                                            const int trainer_rank = 0);

std::vector<std::unique_ptr<lbann_data::LbannPB>>
read_in_prototext_files(const bool master,
                        const std::vector<prototext_fn_triple>& names);

/** @brief attempts to verify the all models are valid, and contain an
 *         optimizer and reader
 */
void verify_prototext(
  const bool master,
  const std::vector<std::unique_ptr<lbann_data::LbannPB>>& models);

} // namespace protobuf_utils

} // namespace lbann
#endif // LBANN_UTILS_PROTOBUF_UTILS_HPP_INCLUDED
