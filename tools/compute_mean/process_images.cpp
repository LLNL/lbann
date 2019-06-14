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
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include "process_images.hpp"
#include "lbann/data_readers/cv_process.hpp"
#include "file_utils.hpp"
#include "mean_image.hpp"


namespace tools_compute_mean {

/**
 * Set up the preprocessor pipeline, pp, consists of cropper, colorizer, and
 * mean_extractor based on the parameter mp. It also returns the transform index
 * of mean_extractor in the pipeline such that mean image can be extracted from
 * the pipeline.
 */
void setup_preprocessor(const params& mp, lbann::cv_process& pp, int& mean_extractor_idx) {

  int transform_idx = 0;

  // Initialize the image processor
  const cropper_params& cp = mp.get_cropper_params();
  unsigned int n_ch = 3u;

  if (cp.m_is_set) { // If cropper parameters are given
    // Setup a cropper
    std::unique_ptr<lbann::cv_cropper> cropper(new(lbann::cv_cropper));
    cropper->set(cp.m_crop_sz.first, cp.m_crop_sz.second, cp.m_rand_center, cp.m_roi_sz);
    pp.add_transform(std::move(cropper));
    transform_idx ++;
  }

  if (mp.to_enable_decolorizer()) { // Set up a decolorizer
    std::unique_ptr<lbann::cv_decolorizer> decolorizer(new(lbann::cv_decolorizer));
    pp.add_transform(std::move(decolorizer));
    n_ch = 1u;
    transform_idx ++;
  }

  if (mp.to_enable_colorizer()) { // Set up a colorizer
    std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
    pp.add_transform(std::move(colorizer));
    n_ch = 3u;
    transform_idx ++;
  }

  if (mp.to_enable_mean_extractor()) { // set up a mean extractor
    mean_extractor_idx = transform_idx;
    std::unique_ptr<lbann::cv_mean_extractor> mean_extractor(new(lbann::cv_mean_extractor));
    if (cp.m_is_set) {
      mean_extractor->set(cp.m_crop_sz.first, cp.m_crop_sz.second, n_ch, mp.get_mean_batch_size());
    } else {
      mean_extractor->set(mp.get_mean_batch_size());
    }
    pp.add_transform(std::move(mean_extractor));
    transform_idx ++;
  }
}


/**
 * This is the main processing loop that crops each image and accumulates its pixel
 * values. Finally, it computes the pixel-wise mean of cropped images and stores it.
 */
bool process_images(const image_list& img_list, const params& mp,
                    const mpi_states& ms, walltimes& wt) {

  int mean_extractor_idx = -1;
  lbann::cv_process pp;
  setup_preprocessor(mp, pp, mean_extractor_idx);

  std::vector<unsigned char> buf;
  std::vector<unsigned char> outbuf;
  const size_t max_img_header_size = 1024;
  const std::vector<std::string>& filenames = img_list.get_image_list();
  const std::string root_data_path = img_list.get_root_data_path();
  const float progress_report_term = ((filenames.size() * mp.get_report_freq() < 1.0)?
                                      1.0 : (filenames.size() * mp.get_report_freq()));
  float next_progress_report_point = progress_report_term;

  double loop_start_time = get_time();

  for (size_t i=0u; i < filenames.size(); ++i) {
    const double step1_start_time = get_time();

    buf.clear();
    // Load an image bytestream into memory
    bool ok = lbann::load_file(root_data_path + filenames[i], buf);
    if (!ok) {
      ms.abort_by_me("Failed to load " + root_data_path + filenames[i]);
    }

    // create a view on a block of bytes
    using InputBuf_T = lbann::cv_image_type<uint8_t>;
    const cv::Mat inbuf(1, buf.size(), InputBuf_T::T(1), &(buf[0]));

    const double step2_start_time = get_time();
    wt.m_load += step2_start_time - step1_start_time;

    // decode the original image
    cv::Mat image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    //std::cout << filenames[i] << ' ' << image.cols << ' ' << image.rows;

    const double step3_start_time = get_time();
    wt.m_decode += step3_start_time - step2_start_time;

    // preprocess the image
    ok = !image.empty() && pp.preprocess(image);

    if (!ok) {
      ms.abort_by_me("Failed to import " + filenames[i]);
    }

    const double step4_start_time = get_time();
    wt.m_preprocess += step4_start_time - step3_start_time;

    if (mp.to_write_cropped()) { // Export the cropped image

      const size_t capacity = lbann::image_data_amount(image) + max_img_header_size;
      if (outbuf.size() < capacity) {
        //std::cout << "bytes reserved for the image: " << image_data_amount(image) << std::endl;
        outbuf.resize(capacity);
      }
      bool ok = !image.empty() && cv::imencode(mp.get_out_ext(), image, outbuf);
      if (!ok) {
        ms.abort_by_me("Failed to write " + filenames[i]);
      }
      std::string ofilename = img_list.get_image_name_with_new_ext(i, mp.get_out_ext());
      write_file(ofilename, outbuf);
      wt.m_write += get_time() - step4_start_time;
    }
    if (ms.is_root()) {
      if (i >= static_cast<size_t>(next_progress_report_point+0.5)) {
        next_progress_report_point += progress_report_term;
        std::string current_progress = "Progress " + std::to_string(static_cast<int>(100.0f * i/filenames.size())) + '%';
        std::cout << current_progress << std::endl;
      }
    }
  }

  wt.m_total = get_time() - loop_start_time;

  if (mp.to_enable_mean_extractor()) {
    // Extract the mean of images
    bool ok = write_mean_image(pp, mean_extractor_idx, ms, img_list.get_out_dir());
    if (!ok) {
      ms.abort("Failed to write mean image.");
    }
  }

  return true;
}

} // end of namespace tools_compute_mean
