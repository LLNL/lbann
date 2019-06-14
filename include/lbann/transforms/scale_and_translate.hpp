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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_TRANSFORMS_SCALE_AND_TRANSLATE_HPP_INCLUDED
#define LBANN_TRANSFORMS_SCALE_AND_TRANSLATE_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/** Scale and Translate data by a constant pair of constants. */
class scale_and_translate : public transform {
public:
  /** Scale_And_Translate all data by scale_and_translate_val. */
  scale_and_translate(float scale_val, float translate_val)
    : transform(), m_scale(scale_val), m_translate(translate_val) {}

  transform* copy() const override { return new scale_and_translate(*this); }

  std::string get_type() const override { return "scale"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;
private:
  /** Amount to scale data by. */
  float m_scale;
  /** Amount to translate data by. */
  float m_translate;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_SCALE_AND_TRANSLATE_HPP_INCLUDED
