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
#ifndef LBANN_DATA_TYPE_WEIGHTS_IMPL_HPP
#define LBANN_DATA_TYPE_WEIGHTS_IMPL_HPP

#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"

namespace lbann {

template <typename TensorDataType>
template <typename ArchiveT>
void
data_type_weights<TensorDataType>
::serialize(ArchiveT& ar)
#if !(defined __CUDACC__)
{
  ar(cereal::base_class<weights>(this),
     CEREAL_NVP(m_values),
     CEREAL_NVP(m_optimizer));
  if constexpr (utils::IsInputArchive<ArchiveT>)
  {
    if (m_optimizer)
      m_optimizer->setup_base(this);
  }
}
#else
;
#endif

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights()
#if !(defined __CUDACC__)
    : data_type_weights(utils::get_current_comm()) {}
#else
  ;
#endif

} // namespace lbann

#endif // LBANN_DATA_TYPE_WEIGHTS_IMPL_HPP
