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

#define LBANN_EXTERNAL_LAYER_INSTANTIATE
#include "lbann/layers/misc/external.hpp"

#include <algorithm>
#include <cstdio>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void external_layer<TensorDataType, Layout, Device>::fp_compute()
{
    printf("FPROP %s\n", Device == El::Device::GPU ? "GPU" : "CPU");
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void external_layer<TensorDataType, Layout, Device>::bp_compute()
{
    printf("BPROP %s\n", Device == El::Device::GPU ? "GPU" : "CPU");
}


#define PROTO_DEVICE(T, Device) \
  extern template class external_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class external_layer<T, data_layout::MODEL_PARALLEL, Device>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate_device.hpp"


} // namespace lbann
