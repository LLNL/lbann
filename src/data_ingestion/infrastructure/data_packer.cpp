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

#include "lbann/data_ingestion/infrastructure/data_packer.hpp"

#include "lbann/utils/exception.hpp"

#include <conduit/conduit_data_type.hpp>
#include <conduit/conduit_node.hpp>
#include <conduit/conduit_utils.hpp>

/* The data_packer class is designed to extract data fields from
 * Conduit nodes and pack them into Hydrogen matrices.
 */
void lbann::data_packer::extract_data_fields_from_samples(
  std::vector<conduit::Node> const& samples,
  std::map<data_field_type, CPUMat*>& input_buffers)
{
  auto const num_samples = samples.size();
  for (auto const& [data_field, X] : input_buffers) {
    LBANN_ASSERT_DEBUG(num_samples <= static_cast<size_t>(X->Width()));
    for (size_t mb_idx = 0UL; mb_idx < num_samples; ++mb_idx) {
      // This call will verify that the extracted sample has the
      // expected size. In particular, the extracted sample's
      // linearized size must equal the height of the input matrix
      // X. The assertion above verifies that the matrix X has
      // sufficient width to accommodate the sample.
      extract_data_field_from_sample(data_field, samples[mb_idx], *X, mb_idx);
    }
  }
}

template <typename OutT, typename SampleT>
static void write_column(OutT* const out,
                         SampleT const* const sample,
                         size_t const sample_size)
{
  std::copy_n(sample, sample_size, out);
}

size_t lbann::data_packer::extract_data_field_from_sample(
  data_field_type const& data_field,
  conduit::Node const& sample,
  CPUMat& X,
  size_t const mb_idx)
{
  std::string const data_id = sample.child(0).name();
  auto const sample_path = conduit::utils::join_path(data_id, data_field);

  // Check to make sure that each Conduit node only has a single
  // sample
  if (sample.number_of_children() != 1)
    LBANN_ERROR("Unsupported number of samples per Conduit node");
  if (!sample.has_path(sample_path))
    LBANN_ERROR("Conduit node has no such path: ", sample_path);
#ifdef LBANN_DEBUG
  if (!sample.is_compact())
    LBANN_WARNING("sample[", data_id, "] does not have a compact layout");
#endif

  conduit::Node const& data_field_node = sample[sample_path];
  size_t const n_elts = data_field_node.dtype().number_of_elements();
  if (n_elts != static_cast<size_t>(X.Height())) {
    LBANN_ERROR(
      "data field ",
      data_field,
      " has ",
      n_elts,
      " elements, but the matrix only has a linearized size (height) of ",
      X.Height());
  }

  auto* const X_column = X.Buffer() + X.LDim() * mb_idx;
  switch (data_field_node.dtype().id()) {
  case conduit::DataType::FLOAT64_ID:
    write_column(X_column, data_field_node.as_float64_ptr(), n_elts);
    break;
  case conduit::DataType::FLOAT32_ID:
    write_column(X_column, data_field_node.as_float32_ptr(), n_elts);
    break;
  case conduit::DataType::INT64_ID:
    write_column(X_column, data_field_node.as_int64_ptr(), n_elts);
    break;
  case conduit::DataType::INT32_ID:
    write_column(X_column, data_field_node.as_int32_ptr(), n_elts);
    break;
  case conduit::DataType::UINT64_ID:
    write_column(X_column, data_field_node.as_uint64_ptr(), n_elts);
    break;
  case conduit::DataType::UINT32_ID:
    write_column(X_column, data_field_node.as_uint32_ptr(), n_elts);
    break;
  default:
    LBANN_ERROR("unknown dtype; not float32/64, int32/64, or uint32/64; dtype "
                "is reported to be: ",
                data_field_node.dtype().name());
  }
  return n_elts;
}

#if 0
size_t data_packer::transform_data_fields(std::map<data_field_type, CPUMat*>& input_buffers,
                                          std::map<data_field_type, transform::transform_pipeline>& input_transformatons)
{
  auto mb_idx = 0;
  for (auto& [data_field, X] : input_buffers) {
    if(input_transformations.find(data_field) == input_transformers.end()) {
      continue;
    }
    mb_idx = 0;
    size_t n_elts = 0;
    for (auto& idx : X.Width()) {
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

}
  size_t transform_data_field(transform::transform_pipeline& transform_pipeline,
                                  CPUMat& X,
                                  //                             int data_id,
                                  int mb_idx)
{
  auto X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  //  auto X_v = create_datum_view(X, mb_idx);
  transform_pipeline.apply(image, X_v, dims);
}
#endif
