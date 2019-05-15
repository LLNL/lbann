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
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <memory> // for dynamic_pointer_cast

namespace lbann {

/// set up a cropper
static void set_cropper(const lbann_data::ImagePreprocessor& pb_preprocessor,
                        const bool master, std::shared_ptr<cv_process>& pp,
                        int& width, int& height) {
  if (pb_preprocessor.has_cropper()) {
    const lbann_data::ImagePreprocessor::Cropper& pb_cropper = pb_preprocessor.cropper();
    if (!pb_cropper.disable()) {
      const std::string cropper_name = ((pb_cropper.name() == "")? "default_cropper" : pb_cropper.name());
      std::unique_ptr<lbann::cv_cropper> cropper(new(lbann::cv_cropper));
      cropper->set_name(cropper_name);
      cropper->set(pb_cropper.crop_width(),
                   pb_cropper.crop_height(),
                   pb_cropper.crop_randomly(),
                   std::make_pair<int,int>(pb_cropper.resized_width(),
                                           pb_cropper.resized_height()),
                   pb_cropper.adaptive_interpolation());
      pp->add_transform(std::move(cropper));
      width = pb_cropper.crop_width();
      height = pb_cropper.crop_height();
      if (master) std::cout << "image processor: " << cropper_name << " cropper is set" << std::endl;
    }
  }
}

/// set up a resizer
static void set_resizer(const lbann_data::ImagePreprocessor& pb_preprocessor,
                        const bool master, std::shared_ptr<cv_process>& pp,
                        int& width, int& height) {
  if (pb_preprocessor.has_resizer()) {
    const lbann_data::ImagePreprocessor::Resizer& pb_resizer = pb_preprocessor.resizer();
    if (!pb_resizer.disable()) {
      const std::string resizer_name = ((pb_resizer.name() == "")? "default_resizer" : pb_resizer.name());
      std::unique_ptr<lbann::cv_resizer> resizer(new(lbann::cv_resizer));
      resizer->set_name(resizer_name);
      resizer->set(pb_resizer.resized_width(),
                   pb_resizer.resized_height(),
                   pb_resizer.adaptive_interpolation());
      pp->add_transform(std::move(resizer));
      width = pb_resizer.resized_width();
      height = pb_resizer.resized_height();
      if (master) std::cout << "image processor: " << resizer_name << " resizer is set" << std::endl;
    }
  }
}

/// set up an augmenter
static void set_augmenter(const lbann_data::ImagePreprocessor& pb_preprocessor,
                          const bool master, std::shared_ptr<cv_process>& pp) {
  if (pb_preprocessor.has_augmenter()) {
    const lbann_data::ImagePreprocessor::Augmenter& pb_augmenter = pb_preprocessor.augmenter();
    if (!pb_augmenter.disable() &&
        (pb_augmenter.horizontal_flip() ||
         pb_augmenter.vertical_flip() ||
         pb_augmenter.rotation() != 0.0 ||
         pb_augmenter.horizontal_shift() != 0.0 ||
         pb_augmenter.vertical_shift() != 0.0 ||
         pb_augmenter.shear_range() != 0.0))
    {
      const std::string augmenter_name = ((pb_augmenter.name() == "")? "default_augmenter" : pb_augmenter.name());
      std::unique_ptr<lbann::cv_augmenter> augmenter(new(lbann::cv_augmenter));
      augmenter->set_name(augmenter_name);
      augmenter->set(pb_augmenter.horizontal_flip(),
                     pb_augmenter.vertical_flip(),
                     pb_augmenter.rotation(),
                     pb_augmenter.horizontal_shift(),
                     pb_augmenter.vertical_shift(),
                     pb_augmenter.shear_range());
      pp->add_transform(std::move(augmenter));
      if (master) std::cout << "image processor: " << augmenter_name << " augmenter is set" << std::endl;
    }
  }
}

/// set up a decolorizer
static void set_decolorizer(const lbann_data::ImagePreprocessor& pb_preprocessor,
                     const bool master, std::shared_ptr<cv_process>& pp, int& channels) {
  if (pb_preprocessor.has_decolorizer()) {
    const lbann_data::ImagePreprocessor::Decolorizer& pb_decolorizer = pb_preprocessor.decolorizer();
    if  (!pb_decolorizer.disable()) {
      const std::string decolorizer_name = ((pb_decolorizer.name() == "")? "default_decolorizer" : pb_decolorizer.name());
      std::unique_ptr<lbann::cv_decolorizer> decolorizer(new(lbann::cv_decolorizer));
      decolorizer->set_name(decolorizer_name);
      decolorizer->set(pb_decolorizer.pick_1ch());
      pp->add_transform(std::move(decolorizer));
      channels = 1;
      if (master) std::cout << "image processor: " << decolorizer_name << " decolorizer is set" << std::endl;
    }
  }
}

/// set up a colorizer
static void set_colorizer(const lbann_data::ImagePreprocessor& pb_preprocessor,
                          const bool master, std::shared_ptr<cv_process>& pp, int& channels) {
  if (pb_preprocessor.has_colorizer()) {
    const lbann_data::ImagePreprocessor::Colorizer& pb_colorizer = pb_preprocessor.colorizer();
    if (!pb_colorizer.disable()) {
      const std::string colorizer_name = ((pb_colorizer.name() == "")? "default_colorizer" : pb_colorizer.name());
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      colorizer->set_name(colorizer_name);
      pp->add_transform(std::move(colorizer));
      channels = 3;
      if (master) std::cout << "image processor: " << colorizer_name << " colorizer is set" << std::endl;
    }
  }
}

static bool has_channel_wise_subtractor(const lbann_data::ImagePreprocessor& pb_preprocessor) {
  if (!pb_preprocessor.has_subtractor()) {
    return false;
  }
  const lbann_data::ImagePreprocessor::Subtractor& pb_subtractor = pb_preprocessor.subtractor();
  return ((pb_subtractor.channel_mean_size() > 0) || (pb_subtractor.channel_stddev_size() > 0))
         && pb_subtractor.image_to_sub().empty() && pb_subtractor.image_to_div().empty();
}

/// set up a subtractor
static void set_subtractor(const lbann_data::ImagePreprocessor& pb_preprocessor,
                           const bool master, std::shared_ptr<cv_process>& pp,
                           const int channels) {
  if (pb_preprocessor.has_subtractor()) {
    const lbann_data::ImagePreprocessor::Subtractor& pb_subtractor = pb_preprocessor.subtractor();
    if  (!pb_subtractor.disable()) {
      const std::string subtractor_name = ((pb_subtractor.name() == "")? "default_subtractor" : pb_subtractor.name());
      std::unique_ptr<lbann::cv_subtractor> subtractor(new(lbann::cv_subtractor));
      subtractor->set_name(subtractor_name);

      bool is_mean_set = false;

      if (!pb_subtractor.image_to_sub().empty()) {
        subtractor->set_mean(pb_subtractor.image_to_sub());
        is_mean_set = true;
      }
      else if (pb_subtractor.channel_mean_size() > 0) {
        const size_t n = pb_subtractor.channel_mean_size();
        if (n != static_cast<size_t>(channels)) {
          throw lbann_exception("Failed to setup subtractor due to inconsistent number of channels.");
        }
        std::vector<lbann::DataType> ch_mean(n);
        for(size_t i = 0u; i < n; ++i) {
          ch_mean[i] = static_cast<lbann::DataType>(pb_subtractor.channel_mean(i));
        }

        subtractor->set_mean(ch_mean);
        is_mean_set = true;
      }

      if (!is_mean_set && master) {
        std::cout << "image processor: " << subtractor_name << " assumes zero mean." << std::endl
                  << "  If this is not the case, provide mean." << std::endl;
      }

      bool is_stddev_set = false;
      if (!pb_subtractor.image_to_div().empty()) {
        subtractor->set_stddev(pb_subtractor.image_to_div());
        is_stddev_set = true;
      }
      else if (pb_subtractor.channel_stddev_size() > 0) {
        const size_t n = pb_subtractor.channel_stddev_size();
        if (n != static_cast<size_t>(channels)) {
          throw lbann_exception("Failed to setup subtractor due to inconsistent number of channels.");
        }
        std::vector<lbann::DataType> ch_stddev(n);
        for(size_t i = 0u; i < n; ++i) {
          ch_stddev[i] = static_cast<lbann::DataType>(pb_subtractor.channel_stddev(i));
        }

        subtractor->set_stddev(ch_stddev);
        is_stddev_set = true;
      }

      pp->add_normalizer(std::move(subtractor));
      if (master) {
        std::cout << "image processor: " << subtractor_name << " subtractor is set for "
                  << (has_channel_wise_subtractor(pb_preprocessor)? "channel-wise" : "pixel-wise")
                  << ' ' << (is_stddev_set? "z-score" : "mean-subtraction") << std::endl;
      }
    }
  }
}

/// set up a sample-wide normalizer
static void set_normalizer(const lbann_data::ImagePreprocessor& pb_preprocessor,
                           const bool master, std::shared_ptr<cv_process>& pp) {
  if (pb_preprocessor.has_normalizer()) {
    const lbann_data::ImagePreprocessor::Normalizer& pb_normalizer = pb_preprocessor.normalizer();
    if (!pb_normalizer.disable()) {
      const std::string normalizer_name = ((pb_normalizer.name() == "")? "default_normalizer" : pb_normalizer.name());
      std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
      normalizer->set_name(normalizer_name);
      normalizer->unit_scale(pb_normalizer.scale());
      normalizer->subtract_mean(pb_normalizer.subtract_mean());
      normalizer->unit_variance(pb_normalizer.unit_variance());
      normalizer->z_score(pb_normalizer.z_score());
      bool ok = pp->add_normalizer(std::move(normalizer));
      if (master && ok) std::cout << "image processor: " << normalizer_name << " normalizer is set" << std::endl;
    }
  }
}


void init_image_preprocessor(const lbann_data::Reader& pb_readme, const bool master,
                             std::shared_ptr<cv_process>& pp, int& width, int& height, int& channels) {
// Currently we set width and height for image_data_reader here considering the transform
// pipeline. image_data_reader reports the final dimension of data to the child layer based
// on these information.
// TODO: However, for composible pipeline, this needs to be automatically determined by each
// cv_process at the setup finalization stage.
  if (!pb_readme.has_image_preprocessor()) return;

  const lbann_data::ImagePreprocessor& pb_preprocessor = pb_readme.image_preprocessor();
  if (pb_preprocessor.disable()) return;

  // data reader name
  const std::string& name = pb_readme.name();
  // final size of image
  width = pb_preprocessor.raw_width();
  height = pb_preprocessor.raw_height();
  if (pb_preprocessor.raw_num_channels() > 0) {
    channels = pb_preprocessor.raw_num_channels();
  }

  if (pb_preprocessor.has_subtractor() && !has_channel_wise_subtractor(pb_preprocessor)) {
    // decolorizer and colorizer are exclusive
    set_decolorizer(pb_preprocessor, master, pp, channels);
    set_colorizer(pb_preprocessor, master, pp, channels);
    // set up a pixel-wise subtractor
    set_subtractor(pb_preprocessor, master, pp, channels);
  }

  set_cropper(pb_preprocessor, master, pp, width, height);
  set_resizer(pb_preprocessor, master, pp, width, height);
  set_augmenter(pb_preprocessor, master, pp);
  if (has_channel_wise_subtractor(pb_preprocessor)) {
    // decolorizer and colorizer are exclusive
    set_decolorizer(pb_preprocessor, master, pp, channels);
    set_colorizer(pb_preprocessor, master, pp, channels);
    // set up a channel-wise subtractor
    set_subtractor(pb_preprocessor, master, pp, channels);
  } else if (!pb_preprocessor.has_subtractor()) {
    // decolorizer/colorizer would have already been applied in the pixel-wise subtractor
    // decolorizer and colorizer are exclusive
    set_decolorizer(pb_preprocessor, master, pp, channels);
    set_colorizer(pb_preprocessor, master, pp, channels);
  }
  set_normalizer(pb_preprocessor, master, pp);

  // create a data reader
  if (name == "imagenet_patches") {
    std::shared_ptr<cv_process_patches> ppp = std::dynamic_pointer_cast<cv_process_patches>(pp);
    if (pb_preprocessor.has_patch_extractor()) {
      const lbann_data::ImagePreprocessor::PatchExtractor& pb_patch_extractor = pb_preprocessor.patch_extractor();
      if (!pb_patch_extractor.disable()) {
        const std::string patch_extractor_name = ((pb_patch_extractor.name() == "")? "default_patch_extractor" : pb_patch_extractor.name());
        lbann::patchworks::patch_descriptor pi;
        pi.set_sample_image(static_cast<unsigned int>(width),
                            static_cast<unsigned int>(height));
        pi.set_size(pb_patch_extractor.patch_width(), pb_patch_extractor.patch_height());
        pi.set_gap(pb_patch_extractor.patch_gap());
        pi.set_jitter(pb_patch_extractor.patch_jitter());
        pi.set_mode_centering(pb_patch_extractor.centering_mode());
        pi.set_mode_chromatic_aberration(pb_patch_extractor.ca_correction_mode());
        pi.set_self_label();
        pi.define_patch_set();
        width = pb_patch_extractor.patch_width();
        height = pb_patch_extractor.patch_height();
        ppp->set_name(patch_extractor_name);
        ppp->set_patch_descriptor(pi);
        if (master) std::cout << "image processor: " << patch_extractor_name << " patch_extractor is set" << std::endl;
      }
    }
  }
}


void init_image_data_reader(const lbann_data::Reader& pb_readme, const lbann_data::DataSetMetaData& pb_metadata, const bool master, generic_data_reader* &reader) {
  // data reader name
  const std::string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();
  // number of labels
  const int n_labels = pb_readme.num_labels();

  std::shared_ptr<cv_process> pp;
  // set up the image preprocessor
  if ((name == "imagenet") || (name == "jag_conduit") ||
      (name == "multihead_siamese") || (name == "mnist_siamese") ||
      (name == "multi_images") || (name == "moving_mnist")) {
    pp = std::make_shared<cv_process>();
  } else if (name == "imagenet_patches") {
    pp = std::make_shared<cv_process_patches>();
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: unknown name for image data reader: "
          << name;
      throw lbann_exception(err.str());
    }
  }

  // final size of image
  int width = 0, height = 0;
  int channels = 0;

  // setup preprocessor
  init_image_preprocessor(pb_readme, master, pp, width, height, channels);

  if (name == "imagenet_patches") {
    std::shared_ptr<cv_process_patches> ppp = std::dynamic_pointer_cast<cv_process_patches>(pp);
    reader = new imagenet_reader_patches(ppp, shuffle);
  } else if (name == "imagenet") {
    reader = new imagenet_reader(pp, shuffle);
  } else if (name == "multihead_siamese") {
    reader = new data_reader_multihead_siamese(pp, pb_readme.num_image_srcs(), shuffle);
  } else if (name == "mnist_siamese") {
    reader = new data_reader_mnist_siamese(pp, shuffle);
  } else if (name == "multi_images") {
    reader = new data_reader_multi_images(pp, shuffle);
  } else if (name == "moving_mnist") {
    reader = new moving_mnist_reader(7, 40, 40, 2);
#ifdef LBANN_HAS_CONDUIT
  } else if (name =="jag_conduit") {
    data_reader_jag_conduit* reader_jag = new data_reader_jag_conduit(pp, shuffle);
    const lbann_data::DataSetMetaData::Schema& pb_schema = pb_metadata.schema();

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
#endif // LBANN_HAS_CONDUIT
  }

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

  // configure the data reader
  if (name == "multi_images") {
    const int n_img_srcs = pb_readme.num_image_srcs();
    data_reader_multi_images* multi_image_dr_ptr
      = dynamic_cast<data_reader_multi_images*>(image_data_reader_ptr);
    if (multi_image_dr_ptr == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " no data_reader_multi_images";
      throw lbann_exception(err.str());
    }
    multi_image_dr_ptr->set_input_params(width, height, channels, n_labels, n_img_srcs);
  } else if(name == "multihead_siamese") {
    const int n_img_srcs = pb_readme.num_image_srcs();
    data_reader_multi_images* multi_image_dr_ptr
      = dynamic_cast<data_reader_multi_images*>(image_data_reader_ptr);
    multi_image_dr_ptr->set_input_params(width, height, channels, n_labels, n_img_srcs);
  } else {
    image_data_reader_ptr->set_input_params(width, height, channels, n_labels);
  }
}


void init_generic_preprocessor(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* reader) {
  if (!pb_readme.has_image_preprocessor()) return;

  const lbann_data::ImagePreprocessor& pb_preprocessor = pb_readme.image_preprocessor();
  if (pb_preprocessor.disable()) return;

  // set up augmenter if necessary
  if (pb_preprocessor.has_augmenter()) {
    const lbann_data::ImagePreprocessor::Augmenter& pb_augmenter = pb_preprocessor.augmenter();
    if (!pb_augmenter.disable() &&
        (pb_augmenter.name() == "") &&
        (pb_augmenter.horizontal_flip() ||
         pb_augmenter.vertical_flip() ||
         pb_augmenter.rotation() != 0.0 ||
         pb_augmenter.horizontal_shift() != 0.0 ||
         pb_augmenter.vertical_shift() != 0.0 ||
         pb_augmenter.shear_range() != 0.0))
    {
      reader->horizontal_flip( pb_augmenter.horizontal_flip() );
      reader->vertical_flip( pb_augmenter.vertical_flip() );
      reader->rotation( pb_augmenter.rotation() );
      reader->horizontal_shift( pb_augmenter.horizontal_shift() );
      reader->vertical_shift( pb_augmenter.vertical_shift() );
      reader->shear_range( pb_augmenter.shear_range() );
      if (master) std::cout << "image processor: augmenter is set" << std::endl;
    } else {
      reader->disable_augmentation();
    }
  }

  // set up the normalizer
  if (pb_preprocessor.has_normalizer()) {
    const lbann_data::ImagePreprocessor::Normalizer& pb_normalizer = pb_preprocessor.normalizer();
    if (!pb_normalizer.disable() &&
        (pb_normalizer.name() == "")) {
      reader->subtract_mean( pb_normalizer.subtract_mean() );
      reader->unit_variance( pb_normalizer.unit_variance() );
      reader->scale( pb_normalizer.scale() );
      reader->z_score( pb_normalizer.z_score() );
      if (master) std::cout << "image processor: normalizer is set" << std::endl;
    }
  }

  if (pb_preprocessor.has_noiser()) {
    const lbann_data::ImagePreprocessor::Noiser& pb_noiser = pb_preprocessor.noiser();
    if (!pb_noiser.disable() &&
        (pb_noiser.name() == "")) {
      reader->add_noise( pb_noiser.factor() );
      if (master) std::cout << "image processor: noiser is set" << std::endl;
    }
  }
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

  // setup preprocessor
  init_generic_preprocessor(pb_readme, master, reader);
}

}
