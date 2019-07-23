////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
#define LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED

#include "lbann/lbann.hpp"
#include <lbann.pb.h>
#include "lbann/proto/factories.hpp"

namespace lbann {

/** @brief Returns true if the Model contains at least one MotifLayer */
bool has_motifs(const lbann_comm& comm, const lbann_data::LbannPB& p);

void expand_motifs(const lbann_comm& comm, lbann_data::LbannPB& pb);

/** @brief Customize the name of the index list
 *
 *  The following options are available
 *   - trainer ID
 *   - model name
 *
 *  The format for the naming convention if the provided name is
 *  \<index list\> is:
 *  @verbatim
    <index list> == <basename>.<extension>
    <model name>_t<ID>_<basename>.<extension> @endverbatim
 */
void customize_data_readers_index_list(const lbann_comm& comm,
                                       lbann_data::LbannPB& p);

/** @brief instantiates one or more generic_data_readers and inserts
 *         them in &data_readers
 */
void init_data_readers(
  lbann_comm *comm,
  const lbann_data::LbannPB& p,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  bool is_shareable_training_data_reader,
  bool is_shareable_testing_data_reader,
  bool is_shareable_validation_data_reader = false);

/** @brief adjusts the number of parallel data readers */
void set_num_parallel_readers(const lbann_comm& comm, lbann_data::LbannPB& p);

/** @brief adjusts the values in p by querying the options db */
void get_cmdline_overrides(const lbann_comm& comm, lbann_data::LbannPB& p);

/** @brief print various params (learn_rate, etc) to cout */
void print_parameters(const lbann_comm& comm, lbann_data::LbannPB& p);

/** @brief prints usage information */
void print_help(const lbann_comm& comm);

/** @brief prints usage information */
void print_help(std::ostream& os);

/** @brief prints prototext file, cmd line, etc to file */
void save_session(const lbann_comm& comm,
                  const int argc, char * const* argv,
                  lbann_data::LbannPB& p);

/** @brief Read prototext from a file into a protobuf message. */
void read_prototext_file(
  const std::string& fn,
  lbann_data::LbannPB& pb,
  const bool master);

/** @brief Write a protobuf message into a prototext file. */
bool write_prototext_file(
  const std::string& fn,
  lbann_data::LbannPB& pb);

/** @brief Parse a space-separated list. */
template <typename T = std::string>
std::vector<T> parse_list(std::string str) {
  std::vector<T> list;
  std::istringstream ss(str);
  for (T entry; ss >> entry;) {
    list.push_back(entry);
  }
  return list;
}

/** @brief Parse a space-separated set. */
template <typename T = std::string>
std::set<T> parse_set(std::string str) {
  std::set<T> set;
  std::istringstream iss(str);
  for (T entry; iss >> entry;) {
    set.insert(entry);
  }
  return set;
}
} // namespace lbann

#endif // LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
