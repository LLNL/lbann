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
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/serialize.hpp"
#include <lbann/layers/misc/dist_embedding.hpp>

#if defined(LBANN_HAS_SHMEM) || defined(LBANN_HAS_NVSHMEM)

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
template <typename ArchiveT>
void dist_embedding_layer<TensorDataType, Layout, Device>::serialize(
  ArchiveT& ar)
{
  using DataTypeLayer = data_type_layer<TensorDataType>;
  ar(::cereal::make_nvp("DataTypeLayer",
                        ::cereal::base_class<DataTypeLayer>(this)),
     CEREAL_NVP(m_num_embeddings),
     CEREAL_NVP(m_embedding_dim),
     CEREAL_NVP(m_sparse_sgd),
     CEREAL_NVP(m_learning_rate),
     CEREAL_NVP(m_barrier_in_forward_prop));
  // Members that aren't serialized
  //   m_embeddings_buffer
  //   m_workspace_buffer_size
  //   m_metadata_buffer_size
  //   m_nb_barrier_request
}

} // namespace lbann

// Manually register the distributed embedding layer since it has many
// permutations of supported data and device types
#include <lbann/macros/common_cereal_registration.hpp>
#define LBANN_COMMA ,
#define PROTO_DEVICE(TYPE, LAYOUT, DEVICE)                                     \
  LBANN_ADD_ALL_SERIALIZE_ETI(::lbann::dist_embedding_layer<                   \
                              TYPE LBANN_COMMA LAYOUT LBANN_COMMA DEVICE>);    \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    ::lbann::dist_embedding_layer<TYPE LBANN_COMMA LAYOUT LBANN_COMMA DEVICE>, \
    "dist_embedding_layer (" #TYPE "," #LAYOUT "," #DEVICE ")");

#ifdef LBANN_HAS_SHMEM
PROTO_DEVICE(float, lbann::data_layout::DATA_PARALLEL, El::Device::CPU)
#ifdef LBANN_HAS_DOUBLE
PROTO_DEVICE(double, lbann::data_layout::DATA_PARALLEL, El::Device::CPU)
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_SHMEM
#ifdef LBANN_HAS_NVSHMEM
PROTO_DEVICE(float, lbann::data_layout::DATA_PARALLEL, El::Device::GPU)
#ifdef LBANN_HAS_DOUBLE
PROTO_DEVICE(double, lbann::data_layout::DATA_PARALLEL, El::Device::GPU)
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_NVSHMEM

LBANN_REGISTER_DYNAMIC_INIT(dist_embedding_layer);

#endif // defined(LBANN_HAS_SHMEM) || defined(LBANN_HAS_NVSHMEM)
