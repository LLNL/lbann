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

#ifndef LBANN_DATA_PACKER_HPP
#define LBANN_DATA_PACKER_HPP

#include "lbann_config.hpp"

#include "lbann/base.hpp"
#include "lbann/data_ingestion/readers/utils/input_data_type.hpp"

// Forward-declare Conduit nodes.
namespace conduit {
class Node;
}

namespace lbann {
namespace data_packer {

/** @brief Copy data fields from Conduit nodes to Hydrogen matrices.
 *
 *  Given a list of samples, each potentially containing multiple
 *  fields, this extracts each of the given fields into an appropriate
 *  matrix. The fields to be extracted are the keys of the
 *  input_buffers map, with their corresponding matrices as their
 *  values. The batch of samples is considered to be ordered, and the
 *  data from the ith sample is written into the ith column of the
 *  corresponding matrix.
 *
 *  @param[in] samples The list of Conduit nodes holding sample data.
 *  @param[in,out] input_buffers A map of data field identifiers to
 *         Hydrogen matrices. The matrices must have the correct size
 *         on input (height equal to the linear size of the respective
 *         data field and width equal to the minibatch size). Any data
 *         in the matrix on input will be overwritten.
 */
void extract_data_fields_from_samples(
  std::vector<conduit::Node> const& samples,
  std::map<data_field_type, CPUMat*>& input_buffers);

/** @brief Copies data from the requested data field into the Hydrogen
 *         matrix.
 *
 *  The data corresponding to the data_field field in the sample
 *  Conduit node is unpacked into the sample_idxth column of the input
 *  matrix, X. The type of the data is converted to
 *  lbann::DataType. The data must be float, double, int32, int64,
 *  uint32, or uint64.
 *
 *  @param[in] data_field The identifier of the field to extract.
 *  @param[in] sample The Conduit node holding the sample data.
 *  @param[in,out] X The matrix into which to put the extracted
 *         sample. Its height must match the linearized sample size,
 *         and its width must be at least sample_idx.
 *  @param[in] sample_idx The column of X into which the sample is placed.
 *  @returns The linearized size of the data field.
 */
size_t extract_data_field_from_sample(data_field_type const& data_field,
                                      conduit::Node const& sample,
                                      CPUMat& X,
                                      size_t sample_idx);

} // namespace data_packer

} // namespace lbann

#endif // LBANN_DATA_PACKER_HPP
