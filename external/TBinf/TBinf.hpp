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
// TBinf.hpp - Tensorboard interface
////////////////////////////////////////////////////////////////////////////////

#ifndef TBINF_HPP_INCLUDED
#define TBINF_HPP_INCLUDED

#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>
#include "event.pb.h"
#include "summary.pb.h"

namespace TBinf {

/**
 * @brief Write data to Tensorboard logging directory in Tensorflow format.
 */
class SummaryWriter {
 public:
  /**
   * @brief Create a new event file in logdir to write to.
   * @param logdir The directory where the event file will be written.
   */
  SummaryWriter(const std::string logdir);
  ~SummaryWriter();

  /**
   * @brief Add a scalar value to the event file.
   * @param tag The tag for this summary.
   * @param value The scalar value.
   * @param step Optional global step.
   */
  void add_scalar(const std::string tag, float value, int64_t step = -1);

  /**
   * @brief Add an image to the event file.
   * @param tag The tag for this summary.
   * @param encoded_img The image to be written.
   * @param dims The dimensions of the image.
   * @param step Optional global step.
   */

 void add_image(const std::string& tag,
                std::string encoded_img,
                const std::vector<size_t>& dims,
                int64_t step = -1);

  /**
   * @brief Add a histogram of values to the event file.
   * @param tag The tag for this summary.
   * @param first Iterator to the first value to add.
   * @param last Iterator past the last value to add.
   * @param step Optional global step.
   */
  void add_histogram(const std::string tag,
                     std::vector<float>::const_iterator first,
                     std::vector<float>::const_iterator last,
                     int64_t step = -1);
  /**
   * @brief Add a histogram based upon buckets to the event file.
   * @param tag The tag for this summary.
   * @param buckets The histogram buckets.
   * @param min The minimum value in the dataset.
   * @param max The maximum value in the dataset.
   * @param num The number of values in the dataset.
   * @param sum The sum of the values in the dataset.
   * @param sqsum The sum of squared values in the dataset.
   * @param step Optional global step.
   */
  void add_histogram(const std::string tag,
                     const std::vector<float> buckets,
                     double min, double max, double num,
                     double sum, double sqsum,
                     int64_t step = -1);
  /** @brief Return the current histogram buckets. */
  const std::vector<double>& get_histogram_buckets() const;
  /** @brief Return the default histogram buckets. */
  static std::vector<double> get_default_histogram_buckets();

  /** @brief Ensure all events are written out. */
  void flush();

 private:
  /**
   * @brief Write a summary to the event file.
   * @param s The summary to write.
   * @param step Optional global step for the event.
   */
  void write_summary_event(tensorflow::Summary *s, int64_t step = -1);

  /**
   * @brief Write an event to the event file.
   * @param e The event to write.
   */
  void write_event(tensorflow::Event& e);

  /** @brief Get current wall time in fractional seconds. */
  double get_time_in_seconds();

  /** @brief Initialize histogram buckets. */
  void init_histogram_buckets();

  /** @brief Current event version. */
  static constexpr const char *EVENT_VERSION = "brain.Event:2";

  /** @brief Filename to write to. */
  std::string filename;

  /** @brief File stream for writing. */
  std::fstream file;

  /** @brief Current histogram buckets. */
  std::vector<double> histogram_buckets;
};

}  // namespace TBinf

#endif  // TBINF_HPP_INCLUDED
