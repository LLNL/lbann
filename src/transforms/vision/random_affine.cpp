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

#include <opencv2/imgproc.hpp>
#include "lbann/transforms/vision/random_affine.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void random_affine::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  auto dst_real = El::Matrix<uint8_t>(utils::get_linearized_size(dims), 1);
  cv::Mat dst = utils::get_opencv_mat(dst_real, dims);
  // Compute the random quantities for the transform.
  // For converting to radians:
  constexpr float pi_rad = 3.14159265358979323846f / 180.0f;
  float angle = 0.0f;
  if (m_rotate_min != 0.0f || m_rotate_max != 0.0f) {
    angle = transform::get_uniform_random(m_rotate_min, m_rotate_max) * pi_rad;
  }
  float translate_x = 0.0f;
  if (m_translate_h != 0.0f) {
    const float dx = dims[2]*m_translate_w;
    translate_x = std::round(transform::get_uniform_random(-dx, dx));
  }
  float translate_y = 0.0f;
  if (m_translate_w != 0.0f) {
    const float dy = dims[1]*m_translate_h;
    translate_y = std::round(transform::get_uniform_random(-dy, dy));
  }
  float scale = 1.0f;
  if (m_scale_min != 0.0f || m_scale_max != 0.0f) {
    scale = transform::get_uniform_random(m_scale_min, m_scale_max);
  }
  float shear = 0.0f;
  if (m_shear_min != 0.0f || m_shear_max != 0.0f) {
    shear = transform::get_uniform_random(m_shear_min, m_shear_max) * pi_rad;
  }
  // Centering matrix:
  const float center_x = dims[2]*0.5f + 0.5f;
  const float center_y = dims[1]*0.5f + 0.5f;
  // Compute the affine transformation matrix: M = T * C * R * S * Sc * C^-1
  // where
  // T = [1 0 translate_x | 0 1 translate_y | 0 0 1]
  // is the translation matrix,
  // C = [1 0 center_x | 0 1 center_y | 0 0 1]
  // is the centering matrix,
  // R = [cos(angle) -sin(angle) 0 | sin(angle) cos(angle) 0 | 0 0 1]
  // is the rotation matrix,
  // S = [1 -sin(shear) 0 | 0 cos(shear) 0 | 0 0 1]
  // is the shear matrix, and
  // Sc = [scale 0 0 | 0 scale 0 | 0 0 1]
  // is the scale matrix.
  // The centering matrix is used to ensure we rotate/shear about the center
  // of the image.
  // What we actually need is the inverse affine map (destination -> source):
  // M^-1 = C * Sc^-1 S^-1 R^-1 C^-1 T^-1.
  // This is a bit ugly to write out fully, but the below is the result, care of
  // Mathematica.
  const float sec_shear_scale = 1.0f / std::cos(shear) / scale;
  float affine_mat[2][3] = {
    {std::cos(angle+shear)*sec_shear_scale, std::sin(angle+shear)*sec_shear_scale, 0.0f},
    {-std::sin(angle)*sec_shear_scale, std::cos(angle)*sec_shear_scale, 0.0f}
  };
  affine_mat[0][2] = affine_mat[0][0]*(-center_x - translate_x)
    + affine_mat[0][1]*(-center_y - translate_y)
    + center_x;
  affine_mat[1][2] = affine_mat[1][0]*(-center_x - translate_x)
    + affine_mat[1][1]*(-center_y - translate_y)
    + center_y;
  cv::Mat cv_affine(2, 3, CV_32F, affine_mat);
  cv::warpAffine(src, dst, cv_affine, dst.size(),
                 cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                 cv::BORDER_REPLICATE);
  data.emplace<uint8_t>(std::move(dst_real));
}

}  // namespace transform
}  // namespace lbann
