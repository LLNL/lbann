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
//
// init_image_data_readers .hpp .cpp - initialize image_data_reader by prototext
////////////////////////////////////////////////////////////////////////////////

#include "lbann/proto/init_image_data_readers.hpp"
#include "lbann/proto/factories.hpp"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <memory> // for dynamic_pointer_cast

namespace lbann {

void init_image_data_reader(const lbann_data::Reader& pb_readme, const lbann_data::DataSetMetaData& pb_metadata, const bool master, generic_data_reader* &reader) {
  // data reader name
  const std::string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();
  // number of labels
  const int n_labels = pb_readme.num_labels();

  // final size of image
  int width = 0, height = 0;
  int channels = 0;

  // Ugly hack for now to extract dimensions.
  for (int i = 0; i < pb_readme.transforms_size(); ++i) {
    auto& trans = pb_readme.transforms(i);
    if (trans.has_center_crop()) {
      height = trans.center_crop().height();
      width = trans.center_crop().width();
    } else if (trans.has_grayscale()) { channels = 1; }
    else if (trans.has_random_crop()) {
      height = trans.random_crop().height();
      width = trans.random_crop().width();
    } else if (trans.has_random_resized_crop()) {
      height = trans.random_resized_crop().height();
      width = trans.random_resized_crop().width();
    } else if (trans.has_random_resized_crop_with_fixed_aspect_ratio()) {
      height = trans.random_resized_crop_with_fixed_aspect_ratio().crop_height();
      width = trans.random_resized_crop_with_fixed_aspect_ratio().crop_width();
    } else if (trans.has_resize()) {
      height = trans.resize().height();
      width = trans.resize().width();
    } else if (trans.has_resized_center_crop()) {
      height = trans.resized_center_crop().crop_height();
      width = trans.resized_center_crop().crop_width();
    }
  }

  if (name == "imagenet") {
    reader = new imagenet_reader(shuffle);
  } else if (name == "multihead_siamese") {
    reader = new data_reader_multihead_siamese(pb_readme.num_image_srcs(), shuffle);
  } else if (name == "moving_mnist") {
    reader = new moving_mnist_reader(7, 40, 40, 2);
  } else if (name =="jag_conduit") {
    data_reader_jag_conduit* reader_jag = new data_reader_jag_conduit(shuffle);
    const lbann_data::DataSetMetaData::Schema& pb_schema = pb_metadata.schema();

    if(height == 0 && pb_schema.image_height() != 0) {
      height = pb_schema.image_height();
    }
    if(width == 0 && pb_schema.image_width() != 0) {
      width = pb_schema.image_width();
    }
    if(channels == 0 && pb_schema.image_num_channels() != 0) {
      channels = pb_schema.image_num_channels();
    }

    if (channels == 0) {
      channels = 1;
    }
    reader_jag->set_image_dims(width, height, channels);

    // Whether to split channels of an image before preprocessing
    if (pb_schema.split_jag_image_channels()) {
      reader_jag->set_split_image_channels();
    } else {
      reader_jag->unset_split_image_channels();
    }

    if(!pb_schema.image_prefix().empty()) {
      reader_jag->set_output_image_prefix(pb_schema.image_prefix());
    }else {
      reader_jag->set_output_image_prefix("/");
    }

    // declare the set of images to use
    std::vector<std::string> image_keys(pb_schema.jag_image_keys_size());

    for (int i=0; i < pb_schema.jag_image_keys_size(); ++i) {
      image_keys[i] = pb_schema.jag_image_keys(i);
    }

    reader_jag->set_image_choices(image_keys);


    using var_t = data_reader_jag_conduit::variable_t;

    // composite independent variable
    std::vector< std::vector<var_t> > independent_type(pb_schema.independent_size());

    for (int i=0; i < pb_schema.independent_size(); ++i) {
      const lbann_data::DataSetMetaData::Schema::JAGDataSlice& slice = pb_schema.independent(i);
      const int slice_size = slice.pieces_size();
      for (int j=0; j < slice_size; ++j) {
        // TODO: instead of using cast, use proper conversion function
        const auto var_type = static_cast<var_t>(slice.pieces(j));
        independent_type[i].push_back(var_type);
      }
    }

    reader_jag->set_independent_variable_type(independent_type);

    // composite dependent variable
    std::vector< std::vector<var_t> > dependent_type(pb_schema.dependent_size());

    for (int i=0; i < pb_schema.dependent_size(); ++i) {
      const lbann_data::DataSetMetaData::Schema::JAGDataSlice& slice = pb_schema.dependent(i);
      const int slice_size = slice.pieces_size();
      for (int j=0; j < slice_size; ++j) {
        // TODO: instead of using cast, use proper conversion function
        const auto var_type = static_cast<var_t>(slice.pieces(j));
        dependent_type[i].push_back(var_type);
      }
    }

    reader_jag->set_dependent_variable_type(dependent_type);

    if(!pb_schema.scalar_prefix().empty()) {
      reader_jag->set_output_scalar_prefix(pb_schema.scalar_prefix());
    }else {
      reader_jag->set_output_scalar_prefix("/");
    }

    // keys of chosen scalar values in jag simulation output
    std::vector<std::string> scalar_keys(pb_schema.jag_scalar_keys_size());

    for (int i=0; i < pb_schema.jag_scalar_keys_size(); ++i) {
      scalar_keys[i] = pb_schema.jag_scalar_keys(i);
    }

    if (scalar_keys.size() > 0u) {
      reader_jag->set_scalar_choices(scalar_keys);
    }

    if(!pb_schema.input_prefix().empty()) {
      reader_jag->set_input_prefix(pb_schema.input_prefix());
    }else {
      reader_jag->set_input_prefix("/");
    }

    // keys of chosen values in jag simulation parameters
    std::vector<std::string> input_keys(pb_schema.jag_input_keys_size());

    for (int i=0; i < pb_schema.jag_input_keys_size(); ++i) {
      input_keys[i] = pb_schema.jag_input_keys(i);
    }

    if (input_keys.size() > 0u) {
      reader_jag->set_input_choices(input_keys);
    }

    // add scalar output keys to filter out
    const int num_scalar_filters = pb_schema.jag_scalar_filters_size();
    for (int i=0; i < num_scalar_filters; ++i) {
      reader_jag->add_scalar_filter(pb_schema.jag_scalar_filters(i));
    }

    // add scalar output key prefixes to filter out by
    const int num_scalar_prefix_filters = pb_schema.jag_scalar_prefix_filters_size();
    for (int i=0; i < num_scalar_prefix_filters; ++i) {
      using prefix_t = lbann::data_reader_jag_conduit::prefix_t;
      const prefix_t pf = std::make_pair(pb_schema.jag_scalar_prefix_filters(i).key_prefix(),
                                         pb_schema.jag_scalar_prefix_filters(i).min_len());
      reader_jag->add_scalar_prefix_filter(pf);
    }

    // add input parameter keys to filter out
    const int num_input_filters = pb_schema.jag_input_filters_size();
    for (int i=0; i < num_input_filters; ++i) {
      reader_jag->add_input_filter(pb_schema.jag_input_filters(i));
    }

    // add scalar output key prefixes to filter out by
    const int num_input_prefix_filters = pb_schema.jag_input_prefix_filters_size();
    for (int i=0; i < num_input_prefix_filters; ++i) {
      using prefix_t = lbann::data_reader_jag_conduit::prefix_t;
      const prefix_t pf = std::make_pair(pb_schema.jag_scalar_prefix_filters(i).key_prefix(),
                                         pb_schema.jag_scalar_prefix_filters(i).min_len());
      reader_jag->add_input_prefix_filter(pf);
    }

    const lbann_data::DataSetMetaData::Normalization& pb_normalization = pb_metadata.normalization();
    // add image normalization parameters
    const int num_image_normalization_params = pb_normalization.jag_image_normalization_params_size();
    for (int i=0; i <  num_image_normalization_params; ++i) {
      using linear_transform_t = lbann::data_reader_jag_conduit::linear_transform_t;
      const linear_transform_t np = std::make_pair(pb_normalization.jag_image_normalization_params(i).scale(),
                                                   pb_normalization.jag_image_normalization_params(i).bias());
      reader_jag->add_image_normalization_param(np);
    }

    // add scalar normalization parameters
    const int num_scalar_normalization_params = pb_normalization.jag_scalar_normalization_params_size();
    for (int i=0; i <  num_scalar_normalization_params; ++i) {
      using linear_transform_t = lbann::data_reader_jag_conduit::linear_transform_t;
      const linear_transform_t np = std::make_pair(pb_normalization.jag_scalar_normalization_params(i).scale(),
                                                   pb_normalization.jag_scalar_normalization_params(i).bias());
      reader_jag->add_scalar_normalization_param(np);
    }

    // add input normalization parameters
    const int num_input_normalization_params = pb_normalization.jag_input_normalization_params_size();
    for (int i=0; i <  num_input_normalization_params; ++i) {
      using linear_transform_t = lbann::data_reader_jag_conduit::linear_transform_t;
      const linear_transform_t np = std::make_pair(pb_normalization.jag_input_normalization_params(i).scale(),
                                                   pb_normalization.jag_input_normalization_params(i).bias());
      reader_jag->add_input_normalization_param(np);
    }

    reader = reader_jag;
    if (master) std::cout << reader->get_type() << " is set" << std::endl;
    return;
  }

  reader->set_transform_pipeline(
    proto::construct_transform_pipeline(pb_readme));

  if (channels == 0) {
    channels = 3;
  }

  auto* image_data_reader_ptr = dynamic_cast<image_data_reader*>(reader);
  if (!image_data_reader_ptr && master) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: invalid image data reader pointer";
    throw lbann_exception(err.str());
  }
  if (master) std::cout << reader->get_type() << " is set" << std::endl;

  image_data_reader_ptr->set_input_params(width, height, channels, n_labels);
}

void init_org_image_data_reader(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* &reader) {
  // data reader name
  const std::string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();

  if (name == "mnist") {
    reader = new mnist_reader(shuffle);
    if (master) std::cout << "mnist_reader is set" << std::endl;
  } else if (name == "cifar10") {
    reader = new cifar10_reader(shuffle);
    if (master) std::cout << "cifar10_reader is set" << std::endl;
  } else if (name == "moving_mnist") {
    reader = new moving_mnist_reader(7, 40, 40, 2);
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: unknown name for image data reader: "
          << name;
      throw lbann_exception(err.str());
    }
  }

  reader->set_transform_pipeline(
    proto::construct_transform_pipeline(pb_readme));
}

}
