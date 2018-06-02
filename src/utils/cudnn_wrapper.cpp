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
#include "lbann/utils/cublas_wrapper.hpp"
#include "lbann/utils/exception.hpp"

#include <iostream>

#include "El.hpp"
#include <unistd.h>

#ifdef LBANN_HAS_CUDNN

namespace lbann
{
namespace cudnn
{

/// It is assumed the number of processes and the number of GPUs on a compute node are equal
cudnn_manager::cudnn_manager(lbann::lbann_comm *_comm,
                             size_t workspace_size,
                             int max_num_gpus,
                             bool nccl_used)
    : comm(_comm), m_workspace_size(workspace_size) {

    // Indicate whether NCCL is used
#ifdef LBANN_HAS_NCCL2
    m_nccl_used = nccl_used;
#else
    if (nccl_used) { LBANN_ERROR("NCCL is requested, but not enabled"); }
    m_nccl_used = false;
#endif

#ifdef HYDROGEN_HAVE_CUB
    // Expand CUB GPU memory pool to contain workspace
    if (m_workspace_size > 0) {
      GPUMat workspace;
      workspace.SetMemoryMode(1);
      workspace.Resize((m_workspace_size + sizeof(DataType) - 1) / sizeof(DataType), 1);
    }
#endif // HYDROGEN_HAVE_CUB

    // Determine number of MPI ranks on current compute node
    // const int rank_in_node = comm->get_rank_in_node();
    const int procs_per_node = comm->get_procs_per_node();

    // Determine number of visible GPUs
    //    CHECK_CUDA(cudaGetDeviceCount(&m_num_visible_gpus));
    m_num_visible_gpus = El::GPUManager::NumDevices();
    if(max_num_gpus >= 0 && max_num_gpus < m_num_visible_gpus) {
        m_num_visible_gpus = max_num_gpus;
    }
    if(m_num_visible_gpus < 1) {
        throw lbann::lbann_exception("cudnn_wrapper: no GPUs found");
    }
    /// It is assumed that the number of processes on this node is equal to the total number of GPUs available

    if(procs_per_node != m_num_visible_gpus){
      std::cout << "cudnn_wrapper: the number of MPI ranks "
                << procs_per_node
                << " is different from than the number of GPUs "
                << m_num_visible_gpus
                << "  available on this node" << std::endl;
    }

    // Construct GPU objects
    m_gpus.push_back(El::GPUManager::Device());
    cudnnHandle_t handle = nullptr;
    FORCE_CHECK_CUDNN(cudnnCreate(&handle));
    FORCE_CHECK_CUDNN(cudnnSetStream(handle, El::GPUManager::Stream()));
    m_handles.assign(1, handle);

    // Get number of GPUs for current MPI rank
    m_num_gpus = m_gpus.size();

    // Make sure LBANN communicator knows GPUs and CUDA streams
    /**  @todo This is a kludge. A better solution would be to
     *   refactor the cuDNN manager and make the LBANN communicator
     *   responsible for GPU management.
     */
    comm->get_gpus() = get_gpus();
    comm->get_cuda_streams() = get_streams();

    /// Setting up for NCCL collective calls
    /// NOTE: For whoever makes changes in this file, please make sure following if statement comes last.
    if(m_nccl_used){
        nccl_setup();
    }
}

cudnn_manager::~cudnn_manager() {

  // Destroy cuDNN handles
  // Use a try-catch block for FORCE_CHECK_{CUDA |CUDNN | CUBLAS} in the
  // destructor -- these could thrown an exception and destructors are
  // considered to be noexcept by default
  try
  {
    for (size_t i=0; i<m_gpus.size(); ++i) {
      if(m_handles[i] != nullptr) {
        FORCE_CHECK_CUDNN(cudnnDestroy(m_handles[i]));
      }
    }
  }
  catch(const std::exception& e)
  {
    std::cerr << "~cudnn_manager: try ... catch " << e.what() << std::endl;
    std::terminate();
  }

  /// NCCL clear
  if(m_nccl_used)
  {
      nccl_destroy();
  }
}

void cudnn_manager::cudnn_manager::synchronize() {
    for(int i=0; i<m_num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(m_gpus[i]));
        CHECK_CUDA(cudaStreamSynchronize(get_stream(i)));
    }
}

void cudnn_manager::cudnn_manager::synchronize_all() {
    for(int i=0; i<m_num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(m_gpus[i]));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

int cudnn_manager::get_num_gpus() const {
    return m_num_gpus;
}

int cudnn_manager::get_num_visible_gpus() const {
    return m_num_visible_gpus;
}

std::vector<int>& cudnn_manager::get_gpus() {
    return m_gpus;
}

const std::vector<int>& cudnn_manager::get_gpus() const {
    return m_gpus;
}

int cudnn_manager::get_gpu(int i) const {
    return m_gpus[i];
}

std::vector<cudaStream_t> cudnn_manager::get_streams() const {
    return std::vector<cudaStream_t>(1, El::GPUManager::Stream());
}

cudaStream_t cudnn_manager::get_stream(int i) const {
    if (i != 0) { LBANN_ERROR("Attempted to access invalid GPU."); }
    return El::GPUManager::Stream();
}

std::vector<cudnnHandle_t>& cudnn_manager::get_handles() {
    return m_handles;
}

const std::vector<cudnnHandle_t>& cudnn_manager::get_handles() const {
    return m_handles;
}

cudnnHandle_t& cudnn_manager::get_handle(int i) {
    return m_handles[i];
}

const cudnnHandle_t& cudnn_manager::get_handle(int i) const {
    return m_handles[i];
}

std::vector<cublasHandle_t> cudnn_manager::get_cublas_handles() const {
    return std::vector<cublasHandle_t>(1, El::GPUManager::cuBLASHandle());
}

cublasHandle_t cudnn_manager::get_cublas_handle(int i) const {
    if (i != 0) { LBANN_ERROR("Attempted to access invalid GPU."); }
    return El::GPUManager::cuBLASHandle();
}

void cudnn_manager::check_error() {
    synchronize();
    for(int i=0; i<m_num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(m_gpus[i]));
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            cudaDeviceReset();
            std::stringstream err;
            err << __FILE__ << " " << __LINE__ << ":: "
                << "CUDA error; err string: " << cudaGetErrorString(status);
            throw lbann::lbann_exception(err.str());
        }
    }
}

void cudnn_manager::nccl_setup() {

#ifdef LBANN_HAS_NCCL2
    if(m_num_gpus != 1){
        char line[1024];
        sprintf(line, "cudnn_manager: the number of GPUs assigned to process is %d; should be 1", m_num_gpus);
        throw lbann::lbann_exception(line);
    }

    /// Create nccl communicators
    int num_gpus_assigned = m_gpus.size();
    m_nccl_comm.resize(num_gpus_assigned);


    int nProcs = comm->get_procs_per_model();
    int myid = comm->get_rank_in_model();
    int total_num_comms = nProcs*num_gpus_assigned;

    ncclUniqueId ncclId;
    if (myid == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
    El::mpi::Comm model_comm = comm->get_model_comm();
    MPI_Comm mpicomm = model_comm.comm;

    /**
       Not sure if we can use Elemental's broadcast for new date type 'ncclUniqeId'.
       For that reason, raw MPI_Bcast is used instead.

       El::mpi::Broadcast(&ncclId, 1, 0, model_comm); */

    /// todo@ check if we can use Elemental's broadcast
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpicomm);

    if (nProcs == 1) {
        int gpuArray = 0;
        NCCLCHECK(ncclCommInitAll(&(m_nccl_comm[0]), 1, &gpuArray));
    }
    else {
        if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupStart());
        for(int i=0; i<num_gpus_assigned; i++){
            FORCE_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
            NCCLCHECK(ncclCommInitRank(&(m_nccl_comm[i]), total_num_comms, ncclId, num_gpus_assigned*myid+i));
        }
        if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupEnd());

    }

#endif // #ifdef LBANN_HAS_NCCL2
}

void cudnn_manager::nccl_destroy() {
#ifdef LBANN_HAS_NCCL2
    int num_gpus_assigned = m_gpus.size();
    for(int i=0; i<num_gpus_assigned; i++){
        ncclCommDestroy(m_nccl_comm[i]);
    }
#endif // #ifdef LBANN_HAS_NCCL2
}

void print_version() {
    std::cout << "cudnnGetVersion() : " << (int)cudnnGetVersion() << " , "
              << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION
              << std::endl;
}

cudnnDataType_t get_cudnn_data_type() {
    switch(sizeof(DataType)) {
    case 2:
        return CUDNN_DATA_HALF;
    case 4:
        return CUDNN_DATA_FLOAT;
    case 8:
        return CUDNN_DATA_DOUBLE;
    default:
        throw lbann::lbann_exception("cudnn_wrapper: invalid data type for cuDNN");
    }
}

void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int num_samples,
                           const std::vector<int>& sample_dims,
                           int sample_stride) {

    // Create tensor descriptor if needed
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }

    // Determine tensor dimensions
    // Note: cuDNN tensors should have at least 4 dimension
    std::vector<int> dims = sample_dims;
    while (dims.size() < 3) {
        dims.insert(dims.begin(), 1);
    }
    dims.insert(dims.begin(), num_samples);

    // Determine tensor strides
    std::vector<int> strides(dims.size());
    strides.back() = 1;
    for(int i = dims.size() - 1; i > 0; --i) {
        strides[i-1] = strides[i] * dims[i];
    }
    strides.front() = std::max(strides.front(), sample_stride);

    // Set cuDNN tensor descriptor
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void set_tensor_cudnn_desc(cudnnTensorDescriptor_t& desc,
                           int height,
                           int width,
                           int leading_dim) {

    // Create tensor descriptor if needed
    if (desc == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    }

    // Determine tensor dimensions and strides
    // Note: cuDNN tensors should have at least 4 dimension
    leading_dim = std::max(height, leading_dim);
    const std::vector<int> dims = {1, 1, width, height};
    const std::vector<int> strides = {width * leading_dim,
                                      width * leading_dim,
                                      leading_dim,
                                      1};

    // Set cuDNN tensor descriptor
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc,
                                           get_cudnn_data_type(),
                                           dims.size(),
                                           dims.data(),
                                           strides.data()));

}

void copy_tensor_cudnn_desc(const cudnnTensorDescriptor_t& src,
                            cudnnTensorDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnDataType_t data_type;
        int num_dims;
        CHECK_CUDNN(cudnnGetTensorNdDescriptor(src,
                                               0,
                                               &data_type,
                                               &num_dims,
                                               nullptr,
                                               nullptr));
        std::vector<int> dims(num_dims), strides(num_dims);
        CHECK_CUDNN(cudnnGetTensorNdDescriptor(src,
                                               num_dims,
                                               &data_type,
                                               &num_dims,
                                               dims.data(),
                                               strides.data()));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(dst,
                                               data_type,
                                               num_dims,
                                               dims.data(),
                                               strides.data()));
    }

}

void copy_kernel_cudnn_desc(const cudnnFilterDescriptor_t& src,
                            cudnnFilterDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnDataType_t data_type;
        cudnnTensorFormat_t format;
        int num_dims;
        CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                               0,
                                               &data_type,
                                               &format,
                                               &num_dims,
                                               nullptr));
        std::vector<int> dims(num_dims);
        CHECK_CUDNN(cudnnGetFilterNdDescriptor(src,
                                               num_dims,
                                               &data_type,
                                               &format,
                                               &num_dims,
                                               dims.data()));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(dst,
                                               data_type,
                                               format,
                                               num_dims,
                                               dims.data()));
    }

}

void copy_convolution_cudnn_desc(const cudnnConvolutionDescriptor_t& src,
                                 cudnnConvolutionDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnConvolutionMode_t mode;
        cudnnDataType_t data_type;
        int num_dims;
        CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                    0,
                                                    &num_dims,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    &mode,
                                                    &data_type));
        std::vector<int> pads(num_dims), strides(num_dims), upscales(num_dims);
        CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
                                                    num_dims,
                                                    &num_dims,
                                                    pads.data(),
                                                    strides.data(),
                                                    upscales.data(),
                                                    &mode,
                                                    &data_type));
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(dst,
                                                    num_dims,
                                                    pads.data(),
                                                    strides.data(),
                                                    upscales.data(),
                                                    mode,
                                                    data_type));
    }

}

void copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
                             cudnnPoolingDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyPoolingDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnPoolingMode_t mode;
        cudnnNanPropagation_t nan_propagation;
        int num_dims;
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                nullptr,
                                                nullptr,
                                                nullptr));
        std::vector<int> dims(num_dims), pads(num_dims), strides(num_dims);
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
        CHECK_CUDNN(cudnnSetPoolingNdDescriptor(dst,
                                                mode,
                                                nan_propagation,
                                                num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
    }

}

void copy_activation_cudnn_desc(const cudnnActivationDescriptor_t& src,
                                cudnnActivationDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyActivationDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnActivationMode_t mode;
        cudnnNanPropagation_t nan_propagation;
        double relu_ceiling;
        CHECK_CUDNN(cudnnGetActivationDescriptor(src,
                                                 &mode,
                                                 &nan_propagation,
                                                 &relu_ceiling));
        CHECK_CUDNN(cudnnSetActivationDescriptor(dst,
                                                 mode,
                                                 nan_propagation,
                                                 relu_ceiling));
    }

}

void copy_lrn_cudnn_desc(const cudnnLRNDescriptor_t& src,
                         cudnnLRNDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreateLRNDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyLRNDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        unsigned n;
        double alpha, beta, k;
        CHECK_CUDNN(cudnnGetLRNDescriptor(src, &n, &alpha, &beta, &k));
        CHECK_CUDNN(cudnnSetLRNDescriptor(dst, n, alpha, beta, k));
    }

}

}// namespace cudnn

}// namespace lbann
#endif // #ifdef LBANN_HAS_CUDNN
