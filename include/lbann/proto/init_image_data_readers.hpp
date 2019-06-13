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

#ifndef LBANN_PROTO_INIT_IMAGE_DATA_READERS_HPP_INCLUDED
#define LBANN_PROTO_INIT_IMAGE_DATA_READERS_HPP_INCLUDED
#include "lbann/proto/proto_common.hpp"
#include "lbann/comm.hpp"

namespace lbann {

extern void init_image_data_reader(const lbann_data::Reader& pb_readme, const lbann_data::DataSetMetaData& pb_metadata, const bool master, generic_data_reader* &reader);
extern void init_org_image_data_reader(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* &reader);

} // namespace lbann

#endif // LBANN_PROTO_INIT_IMAGE_DATA_READERS_HPP_INCLUDED
