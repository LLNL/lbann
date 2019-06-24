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

#include "lbann/proto/factories.hpp"
#include "lbann/transforms/normalize.hpp"
#include "lbann/transforms/sample_normalize.hpp"
#include "lbann/transforms/scale.hpp"
#include "lbann/transforms/vision/adjust_brightness.hpp"
#include "lbann/transforms/vision/adjust_contrast.hpp"
#include "lbann/transforms/vision/adjust_saturation.hpp"
#include "lbann/transforms/vision/center_crop.hpp"
#include "lbann/transforms/vision/colorize.hpp"
#include "lbann/transforms/vision/color_jitter.hpp"
#include "lbann/transforms/vision/cutout.hpp"
#include "lbann/transforms/vision/grayscale.hpp"
#include "lbann/transforms/vision/horizontal_flip.hpp"
#include "lbann/transforms/vision/normalize_to_lbann_layout.hpp"
#include "lbann/transforms/vision/random_affine.hpp"
#include "lbann/transforms/vision/random_crop.hpp"
#include "lbann/transforms/vision/random_resized_crop.hpp"
#include "lbann/transforms/vision/random_resized_crop_with_fixed_aspect_ratio.hpp"
#include "lbann/transforms/vision/resize.hpp"
#include "lbann/transforms/vision/resized_center_crop.hpp"
#include "lbann/transforms/vision/to_lbann_layout.hpp"
#include "lbann/transforms/vision/vertical_flip.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {
namespace proto {

std::unique_ptr<transform::transform> construct_transform(
  const lbann_data::Transform& trans) {
  if (trans.has_normalize()) {
    auto& pb_trans = trans.normalize();
    return make_unique<transform::normalize>(
      parse_list<float>(pb_trans.means()),
      parse_list<float>(pb_trans.stddevs()));
  } else if (trans.has_sample_normalize()) {
    return make_unique<transform::sample_normalize>();
  } else if (trans.has_scale()) {
    return make_unique<transform::scale>(trans.scale().scale());
  } else if (trans.has_center_crop()) {
    auto& pb_trans = trans.center_crop();
    return make_unique<transform::center_crop>(
      pb_trans.height(), pb_trans.width());
  } else if (trans.has_colorize()) {
    return make_unique<transform::colorize>();
  } else if (trans.has_grayscale()) {
    return make_unique<transform::grayscale>();
  } else if (trans.has_horizontal_flip()) {
    return make_unique<transform::horizontal_flip>(
      trans.horizontal_flip().p());
  } else if (trans.has_normalize_to_lbann_layout()) {
    auto& pb_trans = trans.normalize_to_lbann_layout();
    return make_unique<transform::normalize_to_lbann_layout>(
      parse_list<float>(pb_trans.means()),
      parse_list<float>(pb_trans.stddevs()));
  } else if (trans.has_random_affine()) {
    auto& pb_trans = trans.random_affine();
    return make_unique<transform::random_affine>(
      pb_trans.rotate_min(), pb_trans.rotate_max(),
      pb_trans.translate_h(), pb_trans.translate_w(),
      pb_trans.scale_min(), pb_trans.scale_max(),
      pb_trans.shear_min(), pb_trans.shear_max());
  } else if (trans.has_random_crop()) {
    auto& pb_trans = trans.random_crop();
    return make_unique<transform::random_crop>(
      pb_trans.height(), pb_trans.width());
  } else if (trans.has_random_resized_crop()) {
    auto& pb_trans = trans.random_resized_crop();
    // Handle defaults: If one specified, all must be.
    if (pb_trans.scale_min() != 0.0f) {
      return make_unique<transform::random_resized_crop>(
        pb_trans.height(), pb_trans.width(),
        pb_trans.scale_min(), pb_trans.scale_max(),
        pb_trans.ar_min(), pb_trans.ar_max());
    } else {
      return make_unique<transform::random_resized_crop>(
        pb_trans.height(), pb_trans.width());
    }
  } else if (trans.has_random_resized_crop_with_fixed_aspect_ratio()) {
    auto& pb_trans = trans.random_resized_crop_with_fixed_aspect_ratio();
    return make_unique<transform::random_resized_crop_with_fixed_aspect_ratio>(
      pb_trans.height(), pb_trans.width(),
      pb_trans.crop_height(), pb_trans.crop_width());
  } else if (trans.has_resize()) {
    auto& pb_trans = trans.resize();
    return make_unique<transform::resize>(pb_trans.height(), pb_trans.width());
  } else if (trans.has_resized_center_crop()) {
    auto& pb_trans = trans.resized_center_crop();
    return make_unique<transform::resized_center_crop>(
      pb_trans.height(), pb_trans.width(),
      pb_trans.crop_height(), pb_trans.crop_width());
  } else if (trans.has_to_lbann_layout()) {
    return make_unique<transform::to_lbann_layout>();
  } else if (trans.has_vertical_flip()) {
    return make_unique<transform::horizontal_flip>(
      trans.vertical_flip().p());
  } else if (trans.has_adjust_brightness()) {
    return make_unique<transform::adjust_brightness>(
      trans.adjust_brightness().factor());
  } else if (trans.has_adjust_contrast()) {
    return make_unique<transform::adjust_contrast>(
      trans.adjust_contrast().factor());
  } else if (trans.has_adjust_saturation()) {
    return make_unique<transform::adjust_saturation>(
      trans.adjust_saturation().factor());
  } else if (trans.has_color_jitter()) {
    auto& pb_trans = trans.color_jitter();
    return make_unique<transform::color_jitter>(
      pb_trans.min_brightness_factor(), pb_trans.max_brightness_factor(),
      pb_trans.min_contrast_factor(), pb_trans.max_contrast_factor(),
      pb_trans.min_saturation_factor(), pb_trans.max_saturation_factor());
  } else if (trans.has_cutout()) {
    auto& pb_trans = trans.cutout();
    return make_unique<transform::cutout>(
      pb_trans.num_holes(), pb_trans.length());
  }

  LBANN_ERROR("Unknown transform");
  return nullptr;
}

transform::transform_pipeline construct_transform_pipeline(
  const lbann_data::Reader& data_reader) {
  transform::transform_pipeline tp;
  for (int i = 0; i < data_reader.transforms_size(); ++i) {
    tp.add_transform(construct_transform(data_reader.transforms(i)));
  }
  return tp;
}

}  // namespace proto
}  // namespace lbann
