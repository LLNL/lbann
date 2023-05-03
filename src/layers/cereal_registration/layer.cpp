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
#include "lbann/comm.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/distconv_adapter.hpp"
#endif // LBANN_HAS_DISTCONV
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/weights.hpp"
#include <lbann/layers/layer.hpp>

namespace lbann {

template <typename ArchiveT>
void Layer::serialize(ArchiveT& ar)
{
  ar(CEREAL_NVP(m_expected_num_parent_layers),
     CEREAL_NVP(m_expected_num_child_layers),
     CEREAL_NVP(m_frozen),
     CEREAL_NVP(m_name),
     cereal::make_nvp("m_parent_layers", cereal::defer(m_parent_layers)),
     cereal::make_nvp("m_child_layers", cereal::defer(m_child_layers)),
     cereal::make_nvp("m_weights", cereal::defer(m_weights)),
     CEREAL_NVP(m_output_dims_list),
     CEREAL_NVP(m_runs_inplace),
     CEREAL_NVP(m_hint_layer));
  // Members that aren't serialized:
  //   m_model
  //   m_fp_time
  //   m_fp_compute_time
  //   m_bp_time
  //   m_bp_compute_time
  //   m_update_time
  //   m_parallel_strategy
}

} // namespace lbann

#define LBANN_CLASS_NAME Layer
#include <lbann/macros/register_class_with_cereal.hpp>
