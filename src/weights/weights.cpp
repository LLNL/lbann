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

#include "lbann/weights/weights.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/layers.pb.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lbann {

namespace {

/** Get string describing tensor dimensions.
 *  The tensor is stored in a matrix, although there may be multiple
 *  dimensions corresponding to the matrix height and width.
 */
std::string get_dims_string(const std::vector<size_t>& matrix_height_dims,
                            const std::vector<size_t>& matrix_width_dims)
{
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < matrix_height_dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << matrix_height_dims[i];
  }
  ss << ")x(";
  for (size_t i = 0; i < matrix_width_dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << matrix_width_dims[i];
  }
  ss << ")";
  return ss.str();
}

} // namespace

weights::weights() : m_comm(nullptr), m_frozen(false), m_sharded(false)
{

  // Initialize weights name
  static int num_weights = 0;
  m_name = "weights" + std::to_string(num_weights);
  num_weights++;
}

weights::weights(lbann_comm& comm) : weights()
{

  m_comm = &comm;

  setup_default_matrix_distribution();
}

template <typename ArchiveT>
void weights::serialize(ArchiveT& ar)
{
  ar(CEREAL_NVP(m_name), CEREAL_NVP(m_frozen));

  // What about:
  //   m_matrix_height_dims
  //   m_matrix_width_dims
  //   m_matrix_dist
}

description weights::get_description() const
{
  std::ostringstream ss;

  // Construct description object
  description desc(get_name());

  // Dimensions
  const auto& dims = get_dims();
  ss.str(std::string{});
  ss.clear();
  for (size_t i = 0; i < dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << dims[i];
  }
  desc.add("Dimensions", ss.str());

  // Datatype.
  desc.add("Data type", get_datatype_name());

  // Freeze state
  if (is_frozen()) {
    desc.add("Frozen");
  }

  // Sharding state
  if (is_sharded()) {
    desc.add("Sharded");
  }

  // Derived class contribution
  do_augment_description_(desc);

  return desc;
}

// -----------------------------------------------
// Dimension accessors
// -----------------------------------------------

std::vector<size_t> weights::get_dims() const
{
  std::vector<size_t> dims;
  for (const auto& d : get_matrix_width_dims()) {
    dims.push_back(d);
  }
  for (const auto& d : get_matrix_height_dims()) {
    dims.push_back(d);
  }
  return dims;
}
size_t weights::get_size() const { return get_linear_size(get_dims()); }
std::vector<size_t> weights::get_matrix_height_dims() const
{
  return m_matrix_height_dims;
}
std::vector<size_t> weights::get_matrix_width_dims() const
{
  return m_matrix_width_dims;
}
size_t weights::get_matrix_height() const
{
  return m_matrix_height_dims.size() ? get_linear_size(m_matrix_height_dims)
                                     : 1UL;
}
size_t weights::get_matrix_width() const
{
  return m_matrix_width_dims.size() ? get_linear_size(m_matrix_width_dims) : 1;
}
void weights::set_dims(std::vector<size_t> matrix_height_dims,
                       std::vector<size_t> matrix_width_dims)
{
  m_matrix_height_dims = std::move(matrix_height_dims);
  m_matrix_width_dims = std::move(matrix_width_dims);
  do_set_dims_(matrix_height_dims, matrix_width_dims);
}

// -----------------------------------------------
// Matrix distribution accessors
// -----------------------------------------------

void weights::set_values(El::BaseDistMatrix const& values)
{
  if ((values.Height() != get_values_sharded().Height()) ||
      (values.Width() != get_values_sharded().Width())) {
    LBANN_ERROR("Expected matrix size ",
                this->get_matrix_height(),
                "x",
                this->get_matrix_width(),
                "; got a matrix with size ",
                values.Height(),
                "x",
                values.Width());
  }
  El::Copy(values, this->get_values_sharded());
}

El::DistData weights::get_matrix_distribution() const { return m_matrix_dist; }
void weights::set_matrix_distribution(El::DistData dist)
{
  m_matrix_dist = dist;
}

void weights::set_comm(lbann_comm& comm) { m_comm = &comm; }

void weights::setup_default_matrix_distribution()
{
  // Default matrix distribution
  m_matrix_dist.colDist = El::STAR;
  m_matrix_dist.rowDist = El::STAR;
  m_matrix_dist.blockHeight = 1;
  m_matrix_dist.blockWidth = 1;
  m_matrix_dist.colAlign = 0;
  m_matrix_dist.rowAlign = 0;
  m_matrix_dist.colCut = 0;
  m_matrix_dist.rowCut = 0;
  m_matrix_dist.root = 0;
  m_matrix_dist.grid = &(m_comm->get_trainer_grid());
  m_matrix_dist.device = El::Device::CPU;
}

// -----------------------------------------------
// Setup
// -----------------------------------------------

void weights::setup()
{

  // Check that tensor dimensions are valid
  const auto& is_nonpositive = [](size_t d) { return d <= 0; };
  if (std::any_of(m_matrix_height_dims.begin(),
                  m_matrix_height_dims.end(),
                  is_nonpositive) ||
      std::any_of(m_matrix_width_dims.begin(),
                  m_matrix_width_dims.end(),
                  is_nonpositive)) {
    LBANN_ERROR("attempted to setup weights \"",
                this->get_name(),
                "\" with a ",
                get_dims_string(m_matrix_height_dims, m_matrix_width_dims),
                " weights matrix");
  }

  // Derived class setup
  do_setup_();
}

void weights::steal_values(weights& other)
{
  LBANN_ASSERT(this->get_values_sharded().Height() ==
               other.get_values_sharded().Height());
  LBANN_ASSERT(this->get_values_sharded().Width() ==
               other.get_values_sharded().Width());

  if (this->get_values_sharded().DistData() ==
      other.get_values_sharded().DistData())
    this->do_steal_values_(other);
  else
    // Copy things over.
    this->set_values(other.get_values_sharded());
}
} // namespace lbann

#define LBANN_CLASS_NAME weights
#include <lbann/macros/register_class_with_cereal.hpp>
