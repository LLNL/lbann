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

#include "lbann/data_coordinator/data_packer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {
/*
 * The data_packer class is designed to extract data fields from
 * Conduit nodes and pack them into Hydrogen matrices.
 */

size_t data_packer::extract_data_fields_from_samples(std::vector<conduit::Node>& samples,
                                                   std::map<data_field_type, CPUMat*>& input_buffers)
{
  auto mb_idx = 0;
  for (auto& [data_field, X] : input_buffers) {
    mb_idx = 0;
    size_t n_elts = 0;
    for (auto& sample : samples) {
      size_t tmp_n_elts = 0;
      tmp_n_elts = extract_data_field_from_sample(data_field, sample, *X, mb_idx);
      if(n_elts == 0) {
        n_elts = tmp_n_elts;
      }
      if(tmp_n_elts != n_elts) {
        LBANN_ERROR("Unexpected number of elements extracted from the data field ",
                    data_field,
                    " found ", tmp_n_elts,
                    " expected ", n_elts);
      }
      ++mb_idx;
    }
  }
  return mb_idx;
}

size_t data_packer::extract_data_field_from_sample(data_field_type data_field,
                                  conduit::Node& sample,
                                  CPUMat& X,
                                  //                             int data_id,
                                  int mb_idx)
{
  size_t n_elts = 0;
  std::string dtype;
  // Check to make sure that each Conduit node only has a single
  // sample
  if (sample.number_of_children() != 1) {
    LBANN_ERROR("Unsupported number of samples per Conduit node");
  }
  std::string data_id = sample.child(0).name();
  if (!sample.is_compact()) {
    //    sample.print();
    LBANN_WARNING("m_data[",  data_id, "] does not have a compact layout");
  }
#if 0
  if (!sample.is_contiguous()) {
    //    sample.print();
    LBANN_WARNING("m_data[",  data_id, "] does not have a contiguous layout");
  }
  if (sample.data_ptr() == nullptr) {
    LBANN_WARNING("m_data[", data_id, "] does not have a valid data pointer");
  }
  if (sample.contiguous_data_ptr() == nullptr) {
    LBANN_WARNING("m_data[", data_id, "] does not have a valid contiguous data pointer");
  }
#endif
  std::ostringstream ss;
  ss << sample.child(0).name() + "/" << data_field;
  if (!sample.has_path(ss.str())) {
    LBANN_ERROR("no path: ", ss.str());
  }

  conduit::Node const& data_field_node = sample[ss.str()];

  n_elts = data_field_node.dtype().number_of_elements();

  // const void* r;
  dtype = data_field_node.dtype().name();
  if (dtype == "float64") {
    const auto* data = data_field_node.as_float64_ptr();
    // if(data_field_node.dtype().is_compact()) {

    // }
    for (size_t j = 0; j < n_elts; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  else if (dtype == "float32") {
    const auto* data = data_field_node.as_float32_ptr();
    for (size_t j = 0; j < n_elts; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  else if (dtype == "int64") {
    const auto* data = data_field_node.as_int64_ptr();
    for (size_t j = 0; j < n_elts; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  else if (dtype == "int32") {
    const auto* data = data_field_node.as_int32_ptr();
    for (size_t j = 0; j < n_elts; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  else if (dtype == "uint64") {
    const auto* data = data_field_node.as_uint64_ptr();
    for (size_t j = 0; j < n_elts; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  else if (dtype == "uint32") {
    const auto* data = data_field_node.as_uint32_ptr();
    for (size_t j = 0; j < n_elts; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  else {
    LBANN_ERROR("unknown dtype; not float32/64, int32/64, or uint32/64; dtype "
                "is reported to be: ",
                dtype);
  }
  return n_elts;
}

} // namespace lbann
