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
#ifndef LBANN_UNIT_TEST_UTILITIES_TEST_HELPERS_HPP_INCLUDED
#define LBANN_UNIT_TEST_UTILITIES_TEST_HELPERS_HPP_INCLUDED

#include <lbann/data_coordinator/data_coordinator.hpp>
#include <lbann/proto/proto_common.hpp>
#include <lbann/trainers/trainer.hpp>
#include <lbann/utils/argument_parser.hpp>
#include <lbann/utils/options.hpp>

#include <memory>

namespace unit_test {
namespace utilities {

template <typename T>
bool IsValidPtr(std::unique_ptr<T> const& ptr) noexcept
{
  return static_cast<bool>(ptr);
}

template <typename T>
bool IsValidPtr(std::shared_ptr<T> const& ptr) noexcept
{
  return static_cast<bool>(ptr);
}

template <typename T>
bool IsValidPtr(T const* ptr) noexcept
{
  return static_cast<bool>(ptr);
}

/** @brief Return the global LBANN argument parser reset to its default
 *         condition.
 *
 *  With respect to testing, this means that the options have been
 *  added as with lbann::construct_all_options().
 */
inline lbann::default_arg_parser_type& reset_global_argument_parser()
{
  auto& arg_parser = lbann::global_argument_parser();
  arg_parser.clear();
  lbann::construct_all_options();
  return arg_parser;
}

inline void mock_data_reader(lbann::trainer& trainer,
                             const std::vector<El::Int>& sample_size,
                             int num_classes)
{
  lbann::DataReaderMetaData md;
  auto& md_dims = md.data_dims;
  md_dims[lbann::data_reader_target_mode::CLASSIFICATION] = {num_classes};
  md_dims[lbann::data_reader_target_mode::INPUT] = sample_size;

  // Set up the data reader in the data coordinator
  // TODO: This is a bit awkward and can be better refactored
  auto& dc = trainer.get_data_coordinator();
  dc.set_mock_dr_metadata(md);
}

} // namespace utilities
} // namespace unit_test
#endif // LBANN_UNIT_TEST_UTILITIES_TEST_HELPERS_HPP_INCLUDED
