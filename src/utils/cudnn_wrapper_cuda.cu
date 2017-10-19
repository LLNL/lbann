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
//
// cudnn_wrapper .hpp .cpp - cuDNN support - wrapper classes, utility functions
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"

#include "El.hpp"

using namespace cudnn;
using namespace lbann;

namespace cudnn {

namespace {

__global__ void reduce_kernel(DataType *dst, const DataType *src,
                              El::Int len) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= len) return;
  dst[offset] += src[offset];
}

}

void cudnn_manager::allreduce(const std::vector<DataType*>& gpu_data,
                              El::Int height,
                              El::Int width) {
  if (m_num_gpus < 2) {
    return;
  }

  const El::Int buf_len = 1 << 27;
  const El::Int work_len = buf_len * 2; // double buffering
  const El::Int work_len_bytes = work_len * sizeof(DataType);

  std::vector<DataType*> bufs[2];
  for(int i=0; i<m_num_gpus; ++i) {
    if (get_work_space_size(i) < work_len_bytes) {
      set_work_space_size(i, work_len_bytes); 
    }
    bufs[0].push_back(static_cast<DataType*>(get_work_space(i)));
    bufs[1].push_back(static_cast<DataType*>(get_work_space(i)) + buf_len);
  }  


  El::Int total_len = height * width;
  El::Int offset = 0;

  do {
    El::Int len = std::min(total_len - offset, buf_len);
    int sbuf_idx = 0;
    int dbuf_idx = 1;
    for (int j = 0; j < m_num_gpus - 1; ++j) {
      for(int i = 0; i < m_num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(m_gpus[i]));
        int src_dev = i;
        int dst_dev = (i + 1) % m_num_gpus;              
        DataType *src_buf = j == 0 ? gpu_data[src_dev] + offset : bufs[sbuf_idx][src_dev];
        DataType *dst_buf = bufs[dbuf_idx][dst_dev];
        // copy to the next device in the ring
        FORCE_CHECK_CUDA(cudaMemcpyPeerAsync(dst_buf, dst_dev, src_buf, src_dev,
                                             len * sizeof(DataType), get_stream(src_dev)));
      }
      synchronize();
      for(int i = 0; i < m_num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(m_gpus[i]));        
        DataType *dst_buf = bufs[dbuf_idx][i];
        // TODO: use Thrust
        int tb_dim = 256;
        int grid_dim = len / tb_dim + (len % tb_dim ? 1 : 0);
        reduce_kernel<<<grid_dim, tb_dim>>>(gpu_data[i] + offset, dst_buf, len);
      }
      std::swap(sbuf_idx, dbuf_idx);
    }
    offset += len;
  } while (offset < total_len);
  
}

#ifdef __LIB_NCCL
/// Convert DataType to NCCL data type. DataType is either double or float (default).
ncclDataType_t nccl_datatype() {
  switch(sizeof(DataTpe) ) {
    case 8:
      return ncclDouble;
    case 4:
      return ncclFloat;
    case 2:
      return ncclHalf;
    default:
      throw lbann::lbann_exception("cudnn_wrapper: invalid data type for NCCL");
  }
}
#endif


void cudnn_manager::allreduce_nccl(const std::vector<DataType*>& gpu_data,
                              El::Int height,
                              El::Int width) {
#ifdef __LIB_NCCL
#define BUFF_LEN	27
/**
  gpu_data is a vector of pointers, each of which points to a part of
  matrix allocated to GPU memory. Since we assume that one MPI rank is
  assigned to one GPU, the number of element in gpu_data is 1. */

  if (m_num_gpus < 2) {
    return;
  }

  /// It is assumed each MPI rank is assigned to one GPU (that is, m_num_gpus==1)
  int local_rank = comm->get_rank_in_node();
  ncclDataType_t type = nccl_datatype();
  El::Int total_len = height * width;

  DataType *target_buffer;
  CHECK_CUDA(cudaSetDevice(local_rank));

#if 0

  const El::Int buf_len = 1 << BUFF_LEN;
  El::Int offset = 0;

  FORCE_CHECK_CUDA(cudaMalloc((void **) &target_buffer, buf_len*sizeof(DataType)));

  do {
    El::Int len = std::min(total_len - offset, buf_len);

    NCCLCHECK(ncclAllReduce((gpu_data[0]+offset), target_buffer, buf_len, type, ncclSum, m_nccl_comm, get_stream(local_rank)));

    /// Reduction result is stored in target_buffer. Now need to copy the result back to gpu_data.
    FORCE_CHECK_CUDA(cudaMemcpy((gpu_data[0]+offset), target_buffer, buf_len*sizeof(DataType),  cudaMemcpyDeviceToDevice));
    offset += len;
  } while (offset < total_len);

  FORCE_CHECK_CUDA(cudaFree ((void **) &target_buffer));
#else
  FORCE_CHECK_CUDA(cudaMalloc((void **) &target_buffer, total_len*sizeof(DataType)));
  NCCLCHECK(ncclAllReduce(gpu_data[0], target_buffer, total_len, type, ncclSum, m_nccl_comm, get_stream(local_rank)));
  FORCE_CHECK_CUDA(cudaMemcpy(gpu_data[0], target_buffer, total_len*sizeof(DataType),  cudaMemcpyDeviceToDevice));
  FORCE_CHECK_CUDA(cudaFree ((void **) &target_buffer));
#endif

#endif
}


} // namespace cudnn
