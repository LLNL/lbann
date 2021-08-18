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

#ifndef LBANN_INPUT_DATA_TYPE_HPP_INCLUDED
#define LBANN_INPUT_DATA_TYPE_HPP_INCLUDED

#include "lbann/utils/enum_iterator.hpp"
#include <string>

namespace lbann {

using data_field_type = std::string;
#define INPUT_DATA_TYPE_SAMPLES "samples"
#define INPUT_DATA_TYPE_LABELS "labels"
#define INPUT_DATA_TYPE_RESPONSES "responses"

enum class input_data_type {SAMPLES, LABELS, RESPONSES};
using input_data_type_iterator = enum_iterator<input_data_type, input_data_type::SAMPLES, input_data_type::RESPONSES>;
std::string to_string(input_data_type const& idt);

}

#endif // LBANN_INPUT_DATA_TYPE_HPP_INCLUDED
