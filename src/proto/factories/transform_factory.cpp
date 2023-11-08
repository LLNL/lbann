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

#include "lbann/transforms/normalize.hpp"
#include "lbann/transforms/sample_normalize.hpp"
#include "lbann/transforms/scale.hpp"
#include "lbann/transforms/vision/adjust_brightness.hpp"
#include "lbann/transforms/vision/adjust_contrast.hpp"
#include "lbann/transforms/vision/adjust_saturation.hpp"
#include "lbann/transforms/vision/center_crop.hpp"
#include "lbann/transforms/vision/color_jitter.hpp"
#include "lbann/transforms/vision/colorize.hpp"
#include "lbann/transforms/vision/cutout.hpp"
#include "lbann/transforms/vision/grayscale.hpp"
#include "lbann/transforms/vision/horizontal_flip.hpp"
#include "lbann/transforms/vision/normalize_to_lbann_layout.hpp"
#include "lbann/transforms/vision/pad.hpp"
#include "lbann/transforms/vision/random_affine.hpp"
#include "lbann/transforms/vision/random_crop.hpp"
#include "lbann/transforms/vision/random_resized_crop.hpp"
#include "lbann/transforms/vision/random_resized_crop_with_fixed_aspect_ratio.hpp"
#include "lbann/transforms/vision/resize.hpp"
#include "lbann/transforms/vision/resized_center_crop.hpp"
#include "lbann/transforms/vision/to_lbann_layout.hpp"
#include "lbann/transforms/vision/vertical_flip.hpp"

#include "lbann/proto/factories.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/reader.pb.h"
#include "lbann/proto/transforms.pb.h"

namespace {

using factory_type = lbann::generic_factory<
  lbann::transform::transform,
  std::string,
  lbann::generate_builder_type<lbann::transform::transform,
                               google::protobuf::Message const&>,
  lbann::default_key_error_policy>;

void register_default_builders(factory_type& factory)
{
  using namespace lbann::transform;
  factory.register_builder("Normalize", build_normalize_transform_from_pbuf);
  factory.register_builder("SampleNormalize",
                           build_sample_normalize_transform_from_pbuf);
  factory.register_builder("Scale", build_scale_transform_from_pbuf);
#ifdef LBANN_HAS_OPENCV
  factory.register_builder("AdjustBrightness",
                           build_adjust_brightness_transform_from_pbuf);
  factory.register_builder("AdjustContrast",
                           build_adjust_contrast_transform_from_pbuf);
  factory.register_builder("AdjustSaturation",
                           build_adjust_saturation_transform_from_pbuf);
  factory.register_builder("CenterCrop", build_center_crop_transform_from_pbuf);
  factory.register_builder("ColorJitter",
                           build_color_jitter_transform_from_pbuf);
  factory.register_builder("Colorize", build_colorize_transform_from_pbuf);
  factory.register_builder("Cutout", build_cutout_transform_from_pbuf);
  factory.register_builder("Grayscale", build_grayscale_transform_from_pbuf);
  factory.register_builder("HorizontalFlip",
                           build_horizontal_flip_transform_from_pbuf);
  factory.register_builder("NormalizeToLBANNLayout",
                           build_normalize_to_lbann_layout_transform_from_pbuf);
  factory.register_builder("Pad", build_pad_transform_from_pbuf);
  factory.register_builder("RandomAffine",
                           build_random_affine_transform_from_pbuf);
  factory.register_builder("RandomCrop", build_random_crop_transform_from_pbuf);
  factory.register_builder("RandomResizedCrop",
                           build_random_resized_crop_transform_from_pbuf);
  factory.register_builder(
    "RandomResizedCropWithFixedAspectRatio",
    build_random_resized_crop_with_fixed_aspect_ratio_transform_from_pbuf);
  factory.register_builder("Resize", build_resize_transform_from_pbuf);
  factory.register_builder("ResizedCenterCrop",
                           build_resized_center_crop_transform_from_pbuf);
  factory.register_builder("ToLBANNLayout",
                           build_to_lbann_layout_transform_from_pbuf);
  factory.register_builder("VerticalFlip",
                           build_vertical_flip_transform_from_pbuf);
#endif // LBANN_HAS_OPENCV
}

// Manage a global factory
struct factory_manager
{
  factory_type factory_;

  factory_manager() { register_default_builders(factory_); }
};

factory_manager factory_mgr_;
factory_type const& get_transform_factory() noexcept
{
  return factory_mgr_.factory_;
}

} // namespace

std::unique_ptr<lbann::transform::transform>
lbann::proto::construct_transform(const lbann_data::Transform& trans)
{

  auto const& factory = get_transform_factory();
  auto const& msg = protobuf::get_oneof_message(trans, "transform_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg);
}

lbann::transform::transform_pipeline lbann::proto::construct_transform_pipeline(
  const lbann_data::Reader& data_reader_proto)
{
  transform::transform_pipeline tp;
  for (int i = 0; i < data_reader_proto.transforms_size(); ++i) {
    tp.add_transform(construct_transform(data_reader_proto.transforms(i)));
  }
  return tp;
}
