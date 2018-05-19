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

namespace lbann {
namespace cudnn {

namespace {

__global__ void constant_kernel(DataType *data,
                                DataType val,
                                int len) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < len) { data[offset] = val; }
}

__global__ void reduce_kernel(DataType *dst, const DataType *src,
                              int len) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < len) { dst[offset] += src[offset]; }
}

#ifdef LBANN_HAS_NCCL2
__global__ void scale_kernel(DataType *data,
                             const DataType scale,
                             El::Int len) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < len) { data[offset] *= scale; }
}
#endif // LBANN_HAS_NCCL2
}

void cudnn_manager::set_on_gpu(int i,
                               DataType* gpu_data,
                               DataType val,
                               int height,
                               int width) {
  CHECK_CUDA(cudaSetDevice(m_gpus[i]));
  const int len = height * width;
  const int tb_dim = 256;
  const int grid_dim = len / tb_dim + (len % tb_dim ? 1 : 0);
  constant_kernel<<<grid_dim, tb_dim, 0, get_stream(i)>>>(gpu_data, val, len);
}

void cudnn_manager::allreduce_on_gpus(std::vector<DataType*>& gpu_data,
                                      El::Int height,
                                      El::Int width) {
  if (m_num_gpus < 2) {
    return;
  }

  // Determine work space size
  const El::Int work_space_size = get_minimum_work_space_size();
  const El::Int min_work_space_size = 1024;
  if (work_space_size < min_work_space_size) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "insufficient GPU work space "
        << "(requires " << min_work_space_size << " bytes on each GPU, "
        << "but only have " << work_space_size << " bytes)";
    throw lbann_exception(err.str());
  }

  // Setup work buffers
  const El::Int buf_len = work_space_size / (2 * sizeof(DataType));
  std::vector<DataType*> bufs[2];
  for(int i=0; i<m_num_gpus; ++i) {
    DataType* work_space = static_cast<DataType*>(get_work_space(i));
    bufs[0].push_back(work_space);
    bufs[1].push_back(work_space + buf_len);
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
        int src_idx = i;
        int dst_idx = (i + 1) % m_num_gpus;
        int src_dev = m_gpus[src_idx];
        int dst_dev = m_gpus[dst_idx];
        DataType *src_buf = j == 0 ? gpu_data[src_idx] + offset : bufs[sbuf_idx][src_idx];
        DataType *dst_buf = bufs[dbuf_idx][dst_idx];
        // std::cerr << "Copying from device " << src_dev << " to device " << dst_dev << "\n";
        // copy to the next device in the ring
        FORCE_CHECK_CUDA(cudaMemcpyPeerAsync(dst_buf, dst_dev, src_buf, src_dev,
                                             len * sizeof(DataType), get_stream(src_idx)));
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

/// @todo Efficient implementation
void cudnn_manager::global_allreduce_on_gpus(std::vector<DataType*>& gpu_data,
                                             El::Int height,
                                             El::Int width,
                                             El::mpi::Comm comm) {
  if(!is_nccl_used()){
    static Mat cpu_workspace;
    cpu_workspace.Resize(height, width, height);
    allreduce_on_gpus(gpu_data, height, width);
    copy_from_gpu(0, cpu_workspace, gpu_data[0]);
    synchronize();
    El::AllReduce(cpu_workspace, comm);
    broadcast_to_gpus(gpu_data, cpu_workspace);
  } else{
#ifdef LBANN_HAS_NCCL2
    global_allreduce_on_gpus_nccl (gpu_data, height, width);
    synchronize();
#else
    throw lbann_exception("cudnn_manager: NCCL not detected");
#endif // #ifdef LBANN_HAS_NCCL2
  }
}

#ifdef LBANN_HAS_NCCL2
/// Convert DataType to NCCL data type. DataType is either double or float (default).
ncclDataType_t cudnn_manager::nccl_datatype() {
  switch(sizeof(DataType) ) {
    case 8:
      return ncclDouble;
    case 4:
      return ncclFloat;
    case 2:
      return ncclHalf;
    default:
      throw lbann::lbann_exception("cudnn_wrapper_cuda: invalid data type for NCCL");
  }
}

void cudnn_manager::global_allreduce_on_gpus_nccl(std::vector<DataType*>& gpu_data,
                                                  El::Int height,
                                                  El::Int width,
                                                  DataType scale) {


/**
  gpu_data is a vector of pointers, each of which points to a part of
  matrix allocated to GPU memory. Since we assume that one MPI rank is
  assigned to one GPU, the number of element in gpu_data is 1. */

  int num_gpus_assigned = m_gpus.size();

  ncclDataType_t type = nccl_datatype();
  El::Int total_len = height * width;

  if(num_gpus_assigned > 1) ncclGroupStart();
  for(int i = 0; i < num_gpus_assigned; ++i) {
    CHECK_CUDA(cudaSetDevice(m_gpus[i]));
    NCCLCHECK(ncclAllReduce(gpu_data[i], gpu_data[i], total_len, type, ncclSum, m_nccl_comm[i], get_stream(i)));

    /// Apply scaling, if scale != 1
    if(scale != DataType(1)) {
      int tb_dim = 256;
      int grid_dim = total_len/tb_dim + (total_len % tb_dim ? 1 : 0);
      scale_kernel<<<grid_dim, tb_dim>>>(gpu_data[i], scale, total_len);
    }
  }
  if(num_gpus_assigned > 1) ncclGroupEnd();
}
#endif // LBANN_HAS_NCCL2

} // namespace cudnn
} // namespace lbann
