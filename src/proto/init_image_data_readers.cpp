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
// init_image_data_readers .hpp .cpp - initialize image_data_reader by prototext
////////////////////////////////////////////////////////////////////////////////

#include "lbann/proto/init_image_data_readers.hpp"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <memory> // for dynamic_pointer_cast

using namespace lbann;

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

  // set up a subtractor
  if (pb_preprocessor.has_subtractor()) {
    if (pb_preprocessor.has_colorizer()) {
      const lbann_data::ImagePreprocessor::Colorizer& pb_colorizer = pb_preprocessor.colorizer();
      if  (!pb_colorizer.disable()) {
        const std::string colorizer_name = ((pb_colorizer.name() == "")? "default_colorizer" : pb_colorizer.name());
        // If every image in the dataset is a color image, this is not needed
        std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
        colorizer->set_name(colorizer_name);
        pp->add_transform(std::move(colorizer));
        channels = 3;
        if (master) std::cout << "image processor: " << colorizer_name << " colorizer is set" << std::endl;
      }
    }
    const lbann_data::ImagePreprocessor::Subtractor& pb_subtractor = pb_preprocessor.subtractor();
    if  (!pb_subtractor.disable()) {
      const std::string subtractor_name = ((pb_subtractor.name() == "")? "default_subtractor" : pb_subtractor.name());
      // If every image in the dataset is a color image, this is not needed
      std::unique_ptr<lbann::cv_subtractor> subtractor(new(lbann::cv_subtractor));
      subtractor->set_name(subtractor_name);

      bool is_set = false;

      if (!pb_subtractor.image_to_sub().empty()) {
        subtractor->set_mean(pb_subtractor.image_to_sub());
        is_set = true;
      }
      else if (pb_subtractor.channel_mean_size() > 0) {
        const int _width = pb_subtractor.width()? pb_subtractor.width() : width;
        const int _height = pb_subtractor.height()? pb_subtractor.height() : height;

        const size_t n = pb_subtractor.channel_mean_size();
        std::vector<lbann::DataType> ch_mean(n);
        for(size_t i = 0u; i < n; ++i) {
          ch_mean[i] = static_cast<lbann::DataType>(pb_subtractor.channel_mean(i));
        }

        subtractor->set_mean(_width, _height, ch_mean);
        is_set = true;
      }

      if (! pb_subtractor.image_to_div().empty()) {
        subtractor->set_stddev(pb_subtractor.image_to_div());
        is_set = true;
      }
      else if (pb_subtractor.channel_stddev_size() > 0) {
        const int _width = pb_subtractor.width()? pb_subtractor.width() : width;
        const int _height = pb_subtractor.height()? pb_subtractor.height() : height;

        const size_t n = pb_subtractor.channel_stddev_size();
        std::vector<lbann::DataType> ch_stddev(n);
        for(size_t i = 0u; i < n; ++i) {
          ch_stddev[i] = static_cast<lbann::DataType>(pb_subtractor.channel_stddev(i));
        }

        subtractor->set_stddev(_width, _height, ch_stddev);
        is_set = true;
      }

      if (is_set) {
        pp->add_normalizer(std::move(subtractor));
        if (master) std::cout << "image processor: " << subtractor_name << " subtractor is set" << std::endl;
      } else {
        if (master) std::cout << "image processor: " << subtractor_name
                              << " subtractor needs at least either of 'image_to_sub' or 'image_to_div'." << std::endl;
      }
    }
  }

  // set up a cropper
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
  } else { // For backward compatibility. TODO: will be deprecated
    if(pb_preprocessor.crop_first()) {
      std::unique_ptr<lbann::cv_cropper> cropper(new(lbann::cv_cropper));
      cropper->set(pb_preprocessor.crop_width(),
                   pb_preprocessor.crop_height(),
                   pb_preprocessor.crop_randomly(),
                   std::make_pair<int,int>(pb_preprocessor.resized_width(),
                                           pb_preprocessor.resized_height()),
                   pb_preprocessor.adaptive_interpolation());
      pp->add_transform(std::move(cropper));
      if (master) std::cout << "image processor: cropper is set (deprecated syntax)" << std::endl;
    }
  }

  // set up an augmenter
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
  } else { // For backward compatibility. TODO: will be deprecated
    if (!pb_preprocessor.disable_augmentation()) {
      std::unique_ptr<lbann::cv_augmenter> augmenter(new(lbann::cv_augmenter));
      augmenter->set(pb_preprocessor.horizontal_flip(),
                   pb_preprocessor.vertical_flip(),
                   pb_preprocessor.rotation(),
                   pb_preprocessor.horizontal_shift(),
                   pb_preprocessor.vertical_shift(),
                   pb_preprocessor.shear_range());
      pp->add_transform(std::move(augmenter));
      if (master) std::cout << "image processor: augmenter is set (deprecated syntax)" << std::endl;
    }
  }

  // set up a decolorizer
  if (pb_preprocessor.has_decolorizer()) {
    const lbann_data::ImagePreprocessor::Decolorizer& pb_decolorizer = pb_preprocessor.decolorizer();
    if  (!pb_decolorizer.disable()) {
      const std::string decolorizer_name = ((pb_decolorizer.name() == "")? "default_decolorizer" : pb_decolorizer.name());
      // If every image in the dataset is a color image, this is not needed
      std::unique_ptr<lbann::cv_decolorizer> decolorizer(new(lbann::cv_decolorizer));
      decolorizer->set_name(decolorizer_name);
      decolorizer->set(pb_decolorizer.pick_1ch());
      pp->add_transform(std::move(decolorizer));
      channels = 1;
      if (master) std::cout << "image processor: " << decolorizer_name << " decolorizer is set" << std::endl;
    }
  }

  // set up a colorizer
  if (pb_preprocessor.has_colorizer()) {
    const lbann_data::ImagePreprocessor::Colorizer& pb_colorizer = pb_preprocessor.colorizer();
    if  (!pb_colorizer.disable()) {
     if (!pb_preprocessor.has_subtractor()) {
      const std::string colorizer_name = ((pb_colorizer.name() == "")? "default_colorizer" : pb_colorizer.name());
      // If every image in the dataset is a color image, this is not needed
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      colorizer->set_name(colorizer_name);
      pp->add_transform(std::move(colorizer));
      channels = 3;
      if (master) std::cout << "image processor: " << colorizer_name << " colorizer is set" << std::endl;
     }
    }
  } else { // For backward compatibility. TODO: will be deprecated
    if (!pb_preprocessor.no_colorize()) {
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      pp->add_transform(std::move(colorizer));
      channels = 3;
      if (master) std::cout << "image processor: colorizer is set (deprecated syntax)" << std::endl;
    }
  }

  // set up a normalizer
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
      pp->add_normalizer(std::move(normalizer));
      if (master) std::cout << "image processor: " << normalizer_name << " normalizer is set" << std::endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
    normalizer->unit_scale(pb_preprocessor.scale());
    normalizer->subtract_mean(pb_preprocessor.subtract_mean());
    normalizer->unit_variance(pb_preprocessor.unit_variance());
    normalizer->z_score(pb_preprocessor.z_score());
    pp->add_normalizer(std::move(normalizer));
    if (master) std::cout << "image processor: normalizer is set (deprecated syntax)" << std::endl;
  }

  // set up a noiser
  if (pb_preprocessor.has_noiser()) {
    const lbann_data::ImagePreprocessor::Noiser& pb_noiser = pb_preprocessor.noiser();
    if (!pb_noiser.disable()) {
      const std::string noiser_name = ((pb_noiser.name() == "")? "default_noiser" : pb_noiser.name());
/* TODO: implement noiser in opencv
      std::unique_ptr<lbann::cv_noiser> noiser(new(lbann::cv_noiser));
      noiser->set_name(noiser_name);
      noiser->set(pb_noiser.factor());
      pp->add_transform(std::move(noiser));
*/
      if (master) std::cout << "image processor: " << noiser_name << " noiser is not supported yet" << std::endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
/* TODO: implement noiser in opencv
    std::unique_ptr<lbann::cv_noiser> noiser(new(lbann::cv_noiser));
    noiser->set(pb_preprocessor.noise_factor());
    pp->add_transform(std::move(noiser));
*/
    if (master && (pb_preprocessor.noise_factor() > 0.0))
        std::cout << "image processor: noiser is not supported yet (deprecated syntax)" << std::endl;
  }

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


void init_image_data_reader(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* &reader) {
  // data reader name
  const std::string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();
  // number of labels
  const int n_labels = pb_readme.num_labels();

  std::shared_ptr<cv_process> pp;
  // set up the image preprocessor
  if ((name == "imagenet") || (name == "imagenet_single") || (name == "jag_conduit") ||
      (name == "triplet") || (name == "mnist_siamese") || (name == "multi_images")) {
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
  int channels = 3;

  // setup preprocessor
  init_image_preprocessor(pb_readme, master, pp, width, height, channels);

  if (name == "imagenet_patches") {
    std::shared_ptr<cv_process_patches> ppp = std::dynamic_pointer_cast<cv_process_patches>(pp);
    reader = new imagenet_reader_patches(ppp, shuffle);
  } else if (name == "imagenet") {
    reader = new imagenet_reader(pp, shuffle);
  } else if (name == "triplet") {
    reader = new data_reader_triplet(pp, shuffle);
  } else if (name == "mnist_siamese") {
    reader = new data_reader_mnist_siamese(pp, shuffle);
  } else if (name == "multi_images") {
    reader = new data_reader_multi_images(pp, shuffle);
  } else if (name == "imagenet_single") { // imagenet_single
    reader = new imagenet_reader_single(pp, shuffle);
#ifdef LBANN_HAS_CONDUIT
  } else if (name =="jag_conduit") {
    data_reader_jag_conduit* reader_jag = new data_reader_jag_conduit(pp, shuffle);

    reader_jag->set_image_dims(width, height);

    // TODO: parse the list
    const data_reader_jag_conduit::variable_t independent_type
           = static_cast<data_reader_jag_conduit::variable_t>(pb_readme.independent());
    reader_jag->set_independent_variable_type({independent_type});

    const data_reader_jag_conduit::variable_t dependent_type
           = static_cast<data_reader_jag_conduit::variable_t>(pb_readme.dependent());
    reader_jag->set_dependent_variable_type({dependent_type});

    reader = reader_jag;
    if (master) std::cout << reader->get_type() << " is set" << std::endl;
    return;
#endif // LBANN_HAS_CONDUIT
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
  } else { // For backward compatibility. TODO: will be deprecated
    if (!pb_preprocessor.disable_augmentation() &&
        (pb_preprocessor.horizontal_flip() ||
         pb_preprocessor.vertical_flip() ||
         pb_preprocessor.rotation() != 0.0 ||
         pb_preprocessor.horizontal_shift() != 0.0 ||
         pb_preprocessor.vertical_shift() != 0.0 ||
         pb_preprocessor.shear_range() != 0.0)) {
      reader->horizontal_flip( pb_preprocessor.horizontal_flip() );
      reader->vertical_flip( pb_preprocessor.vertical_flip() );
      reader->rotation( pb_preprocessor.rotation() );
      reader->horizontal_shift( pb_preprocessor.horizontal_shift() );
      reader->vertical_shift( pb_preprocessor.vertical_shift() );
      reader->shear_range( pb_preprocessor.shear_range() );
      if (master) std::cout << "image processor: deprecated syntax for augmenter" << std::endl;
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
  } else { // For backward compatibility. TODO: will be deprecated
      reader->subtract_mean( pb_preprocessor.subtract_mean() );
      reader->unit_variance( pb_preprocessor.unit_variance() );
      reader->scale( pb_preprocessor.scale() );
      reader->z_score( pb_preprocessor.z_score() );
      if (master) std::cout << "image processor: deprecated syntax for normalizer" << std::endl;
  }

  if (pb_preprocessor.has_noiser()) {
    const lbann_data::ImagePreprocessor::Noiser& pb_noiser = pb_preprocessor.noiser();
    if (!pb_noiser.disable() &&
        (pb_noiser.name() == "")) {
      reader->add_noise( pb_noiser.factor() );
      if (master) std::cout << "image processor: noiser is set" << std::endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    reader->add_noise( pb_preprocessor.noise_factor() );
    if (master && (pb_preprocessor.noise_factor()>0.0)) std::cout << "image processor: deprecated syntax for noiser" << std::endl;
  }
}


void init_org_image_data_reader(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* &reader) {
  const lbann_data::ImagePreprocessor& pb_preprocessor = pb_readme.image_preprocessor();

  // data reader name
  const std::string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();
  // final size of image. If image_preprocessor is not set, the type-default value
  // (i,e., 0) is used. Then,set_input_params() will not modify the current member value.
  const int width = pb_preprocessor.raw_width();
  const int height = pb_preprocessor.raw_height();

  // number of labels
  const int n_labels = pb_readme.num_labels();

  // TODO: as imagenet_org phases out, and mnist and cifar10 convert to use new
  // imagenet data reader, this function will disappear
  // create data reader
  if (name == "imagenet_org") {
    reader = new imagenet_reader_org(shuffle);
    dynamic_cast<imagenet_reader_org*>(reader)->set_input_params(width, height, 3, n_labels);
    if (master) std::cout << "imagenet_reader_org is set" << std::endl;
  } else if (name == "mnist") {
    reader = new mnist_reader(shuffle);
    if (master) std::cout << "mnist_reader is set" << std::endl;
  } else if (name == "cifar10") {
    reader = new cifar10_reader(shuffle);
    if (master) std::cout << "cifar10_reader is set" << std::endl;
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
