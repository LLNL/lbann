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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_UTILS_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_UTILS_HPP_

namespace lbann {
#if defined LBANN_HAS_CUDNN
namespace dnn_lib {

using namespace cudnn;

namespace internal {

// Simple RAII class that sets the stream on creation, caches the old
// stream, and restores it on the way out.
class StreamManager
{
public:
  StreamManager(cudnnHandle_t handle, cudaStream_t stream) : handle_(handle)
  {
    CHECK_CUDNN(cudnnGetStream(handle_, &old_stream_));
    CHECK_CUDNN(cudnnSetStream(handle_, stream));
  }

  ~StreamManager()
  {
    try {
      if (handle_)
        CHECK_CUDNN(cudnnSetStream(handle_, old_stream_));
    }
    catch (std::exception const& e) {
      std::cerr << "Caught error in ~dnn_lib::StreamManager().\n\n  e.what(): "
                << e.what() << "\n\nCalling std::terminate()." << std::endl;
      std::terminate();
    }
  }
  StreamManager(StreamManager const& other) = delete;
  StreamManager(StreamManager&& other)
    : handle_{other.handle_}, old_stream_{other.old_stream_}
  {
    other.handle_ = nullptr;
    other.old_stream_ = nullptr;
  }
  StreamManager& operator=(StreamManager const& other) = delete;
  StreamManager& operator=(StreamManager&& other) = delete;

  cudnnHandle_t get() const noexcept { return handle_; }

private:
  cudnnHandle_t handle_;
  cudaStream_t old_stream_;
}; // struct StreamManager

inline StreamManager
make_default_handle_manager(El::SyncInfo<El::Device::GPU> const& si)
{
  return StreamManager(get_handle(), si.Stream());
}

} // namespace internal
} // namespace dnn_lib
#endif // defined LBANN_HAS_CUDNN
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_UTILS_HPP_
