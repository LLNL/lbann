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

#ifndef LBANN_DATA_COORDINATOR_METADATA_HPP
#define LBANN_DATA_COORDINATOR_METADATA_HPP

#include <El.hpp>

#include "lbann/utils/enum_iterator.hpp"
#include "lbann_config.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {

// BVE FIXME
// NA - Not applicable, used for input layers that don't produce a second output
enum class data_reader_target_mode
{
  CLASSIFICATION,
  REGRESSION,
  RECONSTRUCTION,
  LABEL_RECONSTRUCTION,
  INPUT,
  NA
};
std::string to_string(data_reader_target_mode m);
/// Map from target modes to dimension maps
using TargetModeDimMap =
  std::unordered_map<data_reader_target_mode, std::vector<El::Int>>;
using data_reader_target_mode_iterator =
  enum_iterator<data_reader_target_mode,
                data_reader_target_mode::CLASSIFICATION,
                data_reader_target_mode::NA>;

/// Map from data_field_type to dimension maps
using data_field_dim_map_type =
  std::unordered_map<data_field_type, std::vector<El::Int>>;

enum class slice_points_mode
{
  INDEPENDENT,
  DEPENDENT,
  NA
};
std::string to_string(const slice_points_mode m);
slice_points_mode slice_points_mode_from_string(const std::string& m);
/// Map from slice points modes to slice points
using SPModeSlicePoints =
  std::unordered_map<slice_points_mode, std::vector<El::Int>>;
using slice_points_mode_iterator = enum_iterator<slice_points_mode,
                                                 slice_points_mode::INDEPENDENT,
                                                 slice_points_mode::NA>;

/// Data structure containing metadata from the data readers
// using DataReaderMetaData = std::pair<TargetModeDimMap,
// TargetModeSlicePoints>;

struct DataReaderMetaData
{
  TargetModeDimMap data_dims;
  SPModeSlicePoints slice_points;

#ifdef LBANN_HAS_DISTCONV
  // Whether tensor shuffle is required. Some data readers such as
  // hyperslab-enabled HDF5 data reader does not require shuffling.
  bool shuffle_required;
#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_DATA_COORDINATOR_METADATA_HPP
