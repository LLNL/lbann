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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"

namespace lbann {

template <El::Device Device>
El::Matrix<DataType, Device>& kfac_block<Device>::get_workspace_matrix(
    const std::string& key, const size_t height, const size_t width) {
  return m_context->get_workspace_matrix(get_name()+" "+key, height, width);
}

template <>
El::SyncInfo<El::Device::CPU> kfac_block<El::Device::CPU>::get_sync_info() {
  return El::SyncInfo<El::Device::CPU>{};
}

#ifdef LBANN_HAS_GPU
template <>
El::SyncInfo<El::Device::GPU> kfac_block<El::Device::GPU>::get_sync_info() {
  return El::gpu::DefaultSyncInfo();
}
#endif // LBANN_HAS_GPU

template class kfac_block<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac_block<El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
