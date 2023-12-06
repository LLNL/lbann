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

#ifndef LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
#define LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED

#include "lbann/data_ingestion/data_reader.hpp"

#define LBANN_ASSERT_MSG_HAS_FIELD(MSG, FIELD)                                 \
  do {                                                                         \
    if (!MSG.has_##FIELD()) {                                                  \
      LBANN_ERROR("No field \"" #FIELD "\" in the given message:\n{\n",        \
                  MSG.DebugString(),                                           \
                  "\n}\n");                                                    \
    }                                                                          \
  } while (false)

// Forward declaration of protobuf classes
namespace lbann_data {
class LbannPB;
class Trainer;
} // namespace lbann_data

namespace lbann {

/** @brief Customize the name of the sample list
 *
 *  The following options are available
 *   - trainer ID
 *   - model name
 *
 *  The format for the naming convention if the provided name is
 *  \<sample list\> is:
 *  @verbatim
    <sample list> == <basename>.<extension>
    <model name>_t<ID>_<basename>.<extension> @endverbatim
 */
void customize_data_readers_sample_list(const lbann_comm& comm,
                                        ::lbann_data::LbannPB& p);

/** @brief instantiates one or more generic_data_readers and inserts
 *         them in &data_readers
 */
void init_data_readers(
  lbann_comm* comm,
  const ::lbann_data::LbannPB& p,
  std::map<execution_mode, generic_data_reader*>& data_readers);

/** @brief adjusts the values in p by querying the options db */
void get_cmdline_overrides(const lbann_comm& comm, ::lbann_data::LbannPB& p);

/** @brief print various params (learn_rate, etc) to cout */
void print_parameters(const lbann_comm& comm,
                      ::lbann_data::LbannPB& p,
                      std::vector<int>& root_random_seeds,
                      std::vector<int>& random_seeds,
                      std::vector<int>& data_seq_random_seeds);

/** @brief prints prototext file, cmd line, etc to file */
void save_session(const lbann_comm& comm,
                  const int argc,
                  char* const* argv,
                  ::lbann_data::LbannPB& p);

/** @brief Read prototext from a file into a protobuf message. */
void read_prototext_file(const std::string& fn,
                         ::lbann_data::LbannPB& pb,
                         const bool master);

/** @brief Read prototext from a string into a protobuf message. */
void read_prototext_string(const std::string& contents,
                           lbann_data::LbannPB& pb,
                           const bool master);

/** @brief Write a protobuf message into a prototext file. */
bool write_prototext_file(const std::string& fn, ::lbann_data::LbannPB& pb);

/** @brief Trim leading and trailing whitespace from a string. */
std::string trim(std::string const& str);

// These functions work on trimmed, nonempty strings
namespace details {

template <typename T>
std::vector<T> parse_list_impl(std::string const& str)
{
#ifdef LBANN_HAS_GPU_FP16
  using ParseType =
    typename std::conditional<std::is_same<T, fp16>::value, float, T>::type;
#else
  using ParseType = T;
#endif
  ParseType entry;
  std::vector<T> list;
  std::istringstream iss(str);
  while (iss.good()) {
    iss >> entry;
    list.emplace_back(std::move(entry));
  }
  return list;
}

template <typename T>
std::set<T> parse_set_impl(std::string const& str)
{
#ifdef LBANN_HAS_GPU_FP16
  using ParseType =
    typename std::conditional<std::is_same<T, fp16>::value, float, T>::type;
#else
  using ParseType = T;
#endif
  ParseType entry;
  std::set<T> set;
  std::istringstream iss(str);
  while (iss.good()) {
    iss >> entry;
    set.emplace(std::move(entry));
  }
  return set;
}

// TODO (trb 07/25/19): we should think about what to do about bad
// input. That is, if a user calls parse_list<int>("one two three"),
// the result is undefined (one test I did gave [0,0,0] and another
// gave [INT_MAX,INT_MAX,INT_MAX]). In most cases in LBANN, I would
// guess that this will result in a logic error further down the
// codepath, but we shouldn't count on it.

} // namespace details

/** @brief Parse a space-separated list. */
template <typename T = std::string>
std::vector<T> parse_list(std::string const& str)
{
  auto trim_str = trim(str);
  if (trim_str.size())
    return details::parse_list_impl<T>(trim_str);
  return {};
}

/** @brief Parse a space-separated set. */
template <typename T = std::string>
std::set<T> parse_set(std::string const& str)
{
  auto trim_str = trim(str);
  if (trim_str.size())
    return details::parse_set_impl<T>(trim_str);
  return {};
}

} // namespace lbann

#endif // LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
