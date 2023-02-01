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
        int m_stride; // mini_batch stride for rank
        int m_group;  // mini batch group for rank
        // Additional buffers on the symmetric heap
        // To do: Remove bufferse - SZ
        Memory<NVSHMEMAllocator> m_output_buffer;
        util::nvshmem::SyncArray m_sync;
      public:
        ScatterNVSHMEM(cudaStream_t stream):m_stream(stream),
                                            m_pid(nvshmem_my_pe()),   
                                            m_np(nvshmem_n_pes()),
                                            m_stride(nvshmem_n_pes()),
                                            m_group(0), 
                                            m_sync(0){}  

        NVHSMEMDevice<DataType> get_for_device(){
          return NVHSMEMDevice<DataType>(m_pid,
                                          m_np,
                                          static_cast<DataType*>(m_output_buffer.get()),
                                          m_sync.get_for_device());
        }

        int get_stride() const {
          return m_stride;
        }

        int get_group() const{
          return m_group;
        }

        void set_stride(int stride){
          m_stride = stride;  
        }

        void set_group(int group){
          m_group = group;
        }

        int get_num_ranks() const{
          return m_np;
        }

        int get_rank() const{
          return m_pid;
        }

        void ensure_buffer(size_t count){        
          size_t cur_size = m_output_buffer.get_size() / sizeof(DataType);

          if (cur_size >= count){
            m_output_buffer.memset(0);
            return ; 
          }

          m_output_buffer.allocate(count * sizeof(DataType));
          m_output_buffer.memset(0);
        }

        void sync(){
          // To do: Replace with non-blocking barrier - SZ
          nvshmemx_barrier_all_on_stream(m_stream);
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
        int m_stride; // mini_batch stride for rank
        int m_group;  // mini batch group for rank
        // Additional buffer on the symmetric heap
        // To do: Remove bufferse - SZ
        Memory<NVSHMEMAllocator> m_output_buffer;
        util::nvshmem::SyncArray m_sync;
      
      public:

        GatherNVSHMEM(cudaStream_t stream):m_stream(stream), 
                                            m_pid(nvshmem_my_pe()),  
                                            m_np(nvshmem_n_pes()),
                                            m_stride(nvshmem_n_pes()),
                                            m_group(0),   
                                            m_sync(0){}  

        NVHSMEMDevice<DataType> get_for_device(){
          return NVHSMEMDevice<DataType>(m_pid,
                                        m_np,
                                        static_cast<DataType*>(m_output_buffer.get()),
                                        m_sync.get_for_device());
        }

        int get_stride() const {
          return m_stride;
        }

        int get_group() const{
          return m_group;
        }

        void set_stride(int stride){
          m_stride = stride;  
        }

        void set_group(int group){
          m_group = group;
        }

        int get_num_ranks() const{
          return m_np;
        }

        int get_rank() const{
          return m_pid;
        }

        void ensure_buffer(size_t count){
          size_t cur_size = m_output_buffer.get_size() / sizeof(DataType);
          if (cur_size >= count){
            m_output_buffer.memset(0);  // 0 out the buffer
            return ; 
          }
          m_output_buffer.allocate(count * sizeof(DataType));
          m_output_buffer.memset(0);
        }

        void sync(){
            // To do: Replace with non-blocking barrier - SZ
            nvshmemx_barrier_all_on_stream(m_stream);
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
