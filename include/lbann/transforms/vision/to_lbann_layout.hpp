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

#ifndef LBANN_TRANSFORMS_TO_LBANN_LAYOUT_HPP_INCLUDED
#define LBANN_TRANSFORMS_TO_LBANN_LAYOUT_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Convert data to LBANN's native data layout.
 * Currently only supports converting from OpenCV layouts.
 * This will also rescale data from [0, 255] to [0, 1].
 */
class to_lbann_layout : public transform {
public:
  transform* copy() const override { return new to_lbann_layout(*this); }

  std::string get_type() const override { return "to_lbann_layout"; }

  bool supports_non_inplace() const override { return true; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;

  void apply(utils::type_erased_matrix& data, CPUMat& out,
             std::vector<size_t>& dims) override;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_TO_LBANN_LAYOUT_HPP_INCLUDED
