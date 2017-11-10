////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
//
// weights_initializer .hpp .cpp - Weights initializer classes
////////////////////////////////////////////////////////////////////////////////

#include "lbann/weights/initializer.hpp"

namespace lbann {

AbsDistMat* weights_initializer::construct_matrix(int height,
                                                  int width,
                                                  El::Distribution col_dist,
                                                  El::Distribution row_dist) const {

  // Construct distributed matrix with desired matrix distribution
  AbsDistMat* weights_matrix = nullptr;
  const El::Grid& grid = m_comm->get_model_grid();
  if (col_dist == El::MC && row_dist == El::MR) {
    weights_matrix = new DistMat(grid);
  }
  if (col_dist == El::STAR && row_dist == El::STAR) {
    weights_matrix = new StarMat(grid);
  }

  // Check that weights has been constructed
  if (weights_matrix == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "could not construct weights matrix with specified distribution "
        << "(col_dist=" << col_dist << ",row_dist=" << row_dist << ")";
    throw lbann_exception(err.str());
  }

  // Resize weights matrix
  weights_matrix->Resize(height, width);

  // Initialize weights matrix entries and return
  initialize_values(*weights_matrix);
  return weights_matrix;

}

void constant_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  if (m_value == DataType(0)) {
    El::Zero(weights_matrix);
  } else {
    El::Fill(weights_matrix, m_value);
  }
}

void uniform_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  const DataType center = (m_max_value + m_min_value) / 2;
  const DataType radius = (m_max_value - m_min_value) / 2;
  uniform_fill(weights_matrix, height, width, center, radius);
}

void normal_initializer::intialize_entries(AbsDistMat& weights_matrix) const {
  gaussian_fill(weights_matrix,
                height, width,
                m_mean, m_standard_deviation);
}

}  // namespace lbann
