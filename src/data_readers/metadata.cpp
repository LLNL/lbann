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

#include "lbann/data_readers/metadata.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

std::string to_string(const data_reader_target_mode m)
{
  switch (m) {
  case data_reader_target_mode::CLASSIFICATION:
    return "classification";
  case data_reader_target_mode::REGRESSION:
    return "regression";
  case data_reader_target_mode::RECONSTRUCTION:
    return "reconstruction";
  case data_reader_target_mode::LABEL_RECONSTRUCTION:
    return "label_reconstruction";
  case data_reader_target_mode::INPUT:
    return "input";
  case data_reader_target_mode::NA:
    return "na";
  default:
    LBANN_ERROR("Invalid data reader target mode specified");
    return "";
  }
}

std::string to_string(const slice_points_mode m)
{
  switch (m) {
  case slice_points_mode::INDEPENDENT:
    return "independent";
  case slice_points_mode::DEPENDENT:
    return "dependent";
  case slice_points_mode::NA:
    return "na";
  default:
    LBANN_ERROR("Invalid slice points mode specified");
    return "";
  }
}

slice_points_mode slice_points_mode_from_string(const std::string& str)
{
  if (str == "independent" || str == "INDEPENDENT") {
    return slice_points_mode::INDEPENDENT;
  }
  if (str == "dependent" || str == "DEPENDENT") {
    return slice_points_mode::DEPENDENT;
  }
  if (str == "na" || str == "NA") {
    return slice_points_mode::NA;
  }
  LBANN_ERROR("Invalid slice points mode specified");
  return slice_points_mode::NA;
}

} // namespace lbann
