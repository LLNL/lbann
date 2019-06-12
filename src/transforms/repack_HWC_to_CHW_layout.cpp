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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/transforms/repack_HWC_to_CHW_layout.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void repack_HWC_to_CHW_layout::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  auto dst = CPUMat(utils::get_linearized_size(dims), 1);
  apply(data, dst, dims);
  data.emplace<DataType>(std::move(dst));
}

void repack_HWC_to_CHW_layout::apply(utils::type_erased_matrix& data, CPUMat& out,
                            std::vector<size_t>& dims) {
  //  std::vector<float> src = data.template get<float>();
  CPUMat &src = data.template get<float>();
  //  cv::Mat src = utils::get_opencv_mat(data, dims);
  // if (!src.isContinuous()) {
  //   // This should not occur, but just in case.
  //   LBANN_ERROR("Do not support non-contiguous OpenCV matrices.");
  // }
  // std::cout << "I am unpacking from dims of " << dims[0] << " channels by " << dims[1] << " x " << dims[2] << " into a matrix of size " << src.Height() << " x " << src.Width()<< " with ldim " << src.LDim() << std::endl;
  // std::cout << "I am unpacking into a dims of " << dims[0] << " channels by " << dims[1] << " x " << dims[2] << " into a matrix of size " << out.Height() << " x " << out.Width() << " with ldim " << out.LDim() << std::endl;
  if (!src.Contiguous()) {
    LBANN_ERROR("RepackHWCtoCHWLayout does not support non-contiguous src.");
  }
  if (!out.Contiguous()) {
    LBANN_ERROR("RepackHWCtoCHWLayout does not support non-contiguous destination.");
  }
  const DataType* __restrict__ src_buf = src.LockedBuffer();
  const size_t out_size = utils::get_linearized_size(dims);
  if (static_cast<size_t>(out.Height() * out.Width()) != out_size) {
    LBANN_ERROR("Transform output does not have sufficient space.");
  }
  DataType* __restrict__ dst_buf = out.Buffer();
  // Pack an interleave multi-channel data structure into a
  // channel-strided data structure
  //  int print = 0;
  const size_t size = dims[1] * dims[2];
  for (size_t row = 0; row < dims[1]; ++row) {
    for (size_t col = 0; col < dims[2]; ++col) {
      int N = dims[0];
      // Multiply by N because there are N channels.
      //      const size_t src_base = N*(row*dims[2] + col);
      const size_t src_base = N*(row + col*dims[1]);
      //const size_t dst_base = row*dims[2] + col;
      const size_t dst_base = row + col*dims[1];
      // if(/*row < 3 && col < 3 ||
      //     */src(src_base,0) != 0 && print < 6) {
      //   print++;
      //   std::cout << "out[0," << dst_base << "]=src[0," << src_base<<"]="<<src(src_base,0)<<std::endl;
      //   std::cout << "out[0," << dst_base+size << "]=src[0," << src_base+1<<"]="<<src(src_base+1,0)<<std::endl;
      //   std::cout << "out[0," << dst_base+2*size << "]=src[0," << src_base+2<<"]="<<src(src_base+2,0)<<std::endl;
      //   std::cout << "out[0," << dst_base+3*size << "]=src[0," << src_base+3<<"]="<<src(src_base+3,0)<< std::endl;
      // }
      switch(N) {
      case 4:
        //out(dst_base + 3*size,0) = src(src_base + 3,0);
        dst_buf[dst_base + 3*size] = src_buf[src_base + 3];
        [[fallthrough]];
      case 3:
        //out(dst_base + 2*size,0) = src(src_base + 2,0);
        dst_buf[dst_base + 2*size] = src_buf[src_base + 2];
        [[fallthrough]];
      case 2:
        //out(dst_base + size,0) = src(src_base + 1,0);
        dst_buf[dst_base + size] = src_buf[src_base + 1];
        [[fallthrough]];
      case 1:
        //out(dst_base,0) = src(src_base,0);
        dst_buf[dst_base] = src_buf[src_base];
        break;
      default:
        LBANN_ERROR("Unsupported number of channels");
      }
    }
  }
}

}  // namespace transform
}  // namespace lbann
