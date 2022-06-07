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

#ifndef LBANN_LAYERS_TRANSFORM_DISTCONV_NVSHMEM_VECTOR_ADDRESSING
#define LBANN_LAYERS_TRANSFORM_DISTCONV_NVSHMEM_VECTOR_ADDRESSING

#if defined(LBANN_HAS_NVSHMEM) && defined(LBANN_HAS_DISTCONV)
#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/nvshmem.hpp"

namespace distconv{
  namespace util{
    /**
     * @brief Array with fixed type and size
     * 
     */
    template <typename T, size_t N>
    struct gpu_array {
      T vals[N];
      __host__ __device__ __forceinline__ size_t size() const {
        return N;
      }

      __host__ __device__ __forceinline__ T& operator[](size_t i){
        return vals[i];
      }

      __host__ __device__ __forceinline__ const T& operator[](size_t i) const{
        return vals[i];
      } 
    };
  } // namespace <distconv::util>
  namespace tensor{
    template <typename DataType>
    struct NVHSMEMDevice{
      int m_pid;
      int m_num_processes;
      DataType *m_buf;
      util::nvshmem::SyncArrayDevice m_sync_device;
      NVHSMEMDevice(int pid,
                    int num_processes, 
                    DataType* buf, 
                    const util::nvshmem::SyncArrayDevice &sync_device):
        m_pid(pid), m_num_processes(num_processes), m_buf(buf), m_sync_device(sync_device){}
    };

    template <typename DataType>
    class ScatterNVSHMEM{
      protected:
        cudaStream_t m_stream;
        int m_pid;
        int m_np;
        Memory<NVSHMEMAllocator> m_buf;
        Memory<NVSHMEMAllocator> m_output_buf;
        util::nvshmem::SyncArray m_sync;
        // Memory<NVSHMEMAllocator> m_native_sync;
      public:
        ScatterNVSHMEM(cudaStream_t stream):m_stream(stream),
                                            m_pid(nvshmem_my_pe()),  // NVSHMEM function 
                                            m_np(nvshmem_n_pes()),   // NVSHMEM function
                                            m_sync(0){}  
      
      NVHSMEMDevice<DataType> get_for_device(){
        return NVHSMEMDevice<DataType>(m_pid,
                                m_np,
                                static_cast<DataType*>(m_buf.get()),
                                m_sync.get_for_device());
      }

      void ensure_buffer(size_t count){        
        size_t cur_size = m_buf.get_size() / sizeof(DataType);
        if (cur_size >= count){
          util::MPIRootPrintStreamInfo() << "Buffer large enough. Continuining";
          return ; 
        }

        util::MPIRootPrintStreamInfo() << "Attempting to allocate NVSHMEM buffer of size " << count * sizeof(DataType); 
        m_buf.allocate(count * sizeof(DataType));
        util::MPIRootPrintStreamInfo() << "Succesfully allocated NVSHMEM buffer of size " << count * sizeof(DataType);
        m_buf.memset(0);
        
      }
      void ensure_output_buffer(size_t count){
        size_t cur_size = m_output_buf.get_size() / sizeof(DataType);
        if (cur_size >= count){
          return ; 
        }
        m_output_buf.allocate(count * sizeof(DataType));
        m_output_buf.memset(0);
      }

      void scatter(const DataType* values, 
                   const DataType* indices,
                   DataType* output,
                   const size_t local_mini_batch_size,
                   const size_t values_rows_size,
                   const size_t values_col_size,
                   const size_t output_rows_size);
    };

    template <typename DataType>
    class GatherNVSHMEM{
      protected:
        cudaStream_t m_stream;
        int m_pid;
        int m_np;
        Memory<NVSHMEMAllocator> m_buf;
        Memory<NVSHMEMAllocator> m_output_buf;
        util::nvshmem::SyncArray m_sync;
        // Memory<NVSHMEMAllocator> m_native_sync;
      
      public:
        GatherNVSHMEM(cudaStream_t stream):m_stream(stream), 
                                           m_pid(nvshmem_my_pe()),  // NVSHMEM function
                                           m_np(nvshmem_n_pes()),   // NVSHMEM function
                                           m_sync(0){}  
        ~GatherNVSHMEM() = default;
      NVHSMEMDevice<DataType> get_for_device(){
        return NVHSMEMDevice<DataType>(m_pid,
                                       m_np,
                                       static_cast<DataType*>(m_buf.get()),
                                       m_sync.get_for_device());
      }

      void ensure_buffer(size_t count){
        size_t cur_size = m_buf.get_size() / sizeof(DataType);
        if (cur_size >= count){
          util::MPIPrintStreamInfo() << "Buffer large enough. Continuining";
          return ; 
        }

        util::MPIPrintStreamInfo() << "Allocating NVSHMEM buffer of size " << count; 
        m_buf.allocate(count * sizeof(DataType));
        m_buf.memset(0);
      }
      void gather(const DataType* values,
                  const DataType* indices,
                  DataType* output,
                  const size_t local_mini_batch_size,
                  const size_t values_rows_size,
                  const size_t values_cols_size,
                  const size_t output_rows_size);
    };
  
  } // namespace distconv::tensor
} // namespace distconv
#endif // LBANN_HAS_NVSHMEM
#endif //LBANN_LAYERS_TRANSFORM_DISTCONV_NVSHMEM_VECTOR_ADDRESSING
