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

#ifndef LBANN_LAYERS_LEARNING_DISTCONV_SCATTER_NVSHMEM
#define LBANN_LAYERS_LEARNING_DISTCONV_SCATTER_NVSHMEM
#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"

#ifdef DISTCONV_HAS_NVSHMEM

#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/nvshmem.hpp"

namespace distconv{
  namespace tensor{
    template <typename DataType>
    struct ScatterNVHSMEMDevice{
      int m_pid;
      int m_num_processes;
      DataType *m_buf;
      util::nvshmem::SyncArrayDevice m_sync;
      ScatterNVHSMEMDevice(int pid,
                           int num_processes, 
                           DataType* buf, 
                           const util::nvshmem::SyncArrayDevice &sync):
        m_pid(pid), m_num_processes(num_processes), m_buf(buf) m_sync(sync){}
    };

    template <typename DataType>
    class ScatterNVSHMEM{
      protected:
        cudaStream_t m_stream;
        int m_pid;
        int m_np;
        Memory<NVSHMEMAllocator> m_buf;
        util::nvshmem::SyncArray m_sync;
        Memory<NVSHMEMAllocator> m_native_sync;
      
      template <typename T>
      ScatterNVHSMEMDevice<T> get_for_device(){
        return ScatterNVHSMEMDevice<T>(m_pid,
                                       m_np,
                                       static_cast<T*>(m_buf.get()),
                                       m_sync.get_for_device());
      }

      void buffer_init(size_t count){
        size_t cur_size = m_buf.get_size() / sizeof(DataType);
        if (cur_size >= count){
          util::MPIPrintStreamInfo() << "Buffer large enough. Continuining";
          return ; 
        }

        util::MPIPrintStreamInfo() << "Allocating NVSHMEM buffer of size " << count; 
        m_buf.allocate(count * sizeof(DataType));
        m_buf.memset(0);
      }
    };
  } // namespace distconv::tensor
} // namespace distconv
#endif // DISTCONV_HAS_NVSHMEM
#endif //LBANN_LAYERS_LEARNING_DISTCONV_SCATTER_NVSHMEM