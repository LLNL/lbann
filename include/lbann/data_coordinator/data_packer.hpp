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

#ifndef LBANN_DATA_PACKER_HPP
#define LBANN_DATA_PACKER_HPP

#include "lbann_config.hpp"

#include "lbann/base.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"
#include "conduit/conduit_node.hpp"

namespace lbann {

namespace data_packer {

  size_t extract_data_fields_from_samples(std::vector<conduit::Node>& samples,
                                          std::map<data_field_type, CPUMat*>& input_buffers);

  /** Copies data from the requested data field into the Hydrogen matrix.
   */
  size_t extract_data_field_from_sample(data_field_type data_field,
                                        conduit::Node& sample,
                                        CPUMat& X,
                                        int mb_idx);
} // namespace data_packer

} // namespace lbann

#endif // LBANN_DATA_PACKER_HPP
