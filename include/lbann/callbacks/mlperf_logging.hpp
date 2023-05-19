////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
// mlperf_logging .hpp .cpp - Prints mlperf compliant benchmark logs
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_MLPERF_LOGGING_HPP_INCLUDED
#define LBANN_CALLBACKS_MLPERF_LOGGING_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <h2/utils/Logger.hpp>

namespace lbann_data {
class Callback;
}

namespace lbann {
namespace callback {

/** @class mlperf_logging
 *  @brief Callback to print mlperf compliant benchmark logs
 */
class mlperf_logging : public callback_base {

public:

  enum class event_type {
    TIME_POINT,
    INT_START,
    INT_END,
  };

public:

  /** @brief mlperf_logging Constructor.
   *  @param string sub_benchmark Name of benchmark.
   *  @param string sub_org Name of submission organization (Default: LBANN)
   *  @param string sub_division Division of benchmark suite (open or closed)
   *  @param string sub_status Submission status (onprem, cloud, or preview)
   *  @param string sub_platform Submission platform/hardware
   */
  mlperf_logging(std::string sub_benchmark, std::string sub_org,
                 std::string sub_division, std::string sub_status,
                 std::string sub_platform)
    : callback_base(/*batch_interval=*/1),
      m_sub_benchmark{sub_benchmark.size() ?
                      std::move(sub_benchmark) :
                      std::string("UNKNOWN_SUBMISSION_BENCHMARK")},
      m_sub_org{sub_org.size() ?
                std::move(sub_org) :
                std::string("LBANN")},
      m_sub_division{sub_division.size() ?
                     std::move(sub_division) :
                     std::string("UNKNOWN_SUBMISSION_DIVISION")},
      m_sub_status{sub_status.size() ?
                   std::move(sub_status) :
                   std::string("UNKNOWN_SUBMISSION_STATUS")},
      m_sub_platform{sub_platform.size() ?
                     std::move(sub_platform) :
                     std::string("UNKNOWN_SUBMISSION_PLATFORM")}
  {}

  /** @brief Copy interface */
  mlperf_logging* copy() const override {
    return new mlperf_logging(*this);
  }

  /** @brief Return name of callback */
  std::string name() const override { return "mlperf_logging"; }

  /** @brief Push mlperf formatted log string to stream object.
   *  @param ostringstream os Stores log strings.
   *  @param event_type et Type of mlperf style event.
   *  @param string key Mlperf log key.
   *  @param T value Mlperf log value.
   *  @param char const* file Current file name.
   *  @param size_t line File line number.
   *  @param double epoch Current epoch number.
   */
  template <typename T>
  void print(std::ostringstream& os, mlperf_logging::event_type et,
             std::string key, T value, char const* file, size_t line,
             double epoch = -1) const;

  void setup(model *m) override;
  void on_setup_end(model *m) override;
  void on_epoch_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_train_begin(model *m) override;
  void on_train_end(model *m) override;
  void on_batch_evaluate_begin(model *m) override;
  void on_batch_evaluate_end(model *m) override;

private:

  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** @brief Populate log with mlperf event type.
   *  @param ostringstream os Stores log string.
   *  @param event_type et Type of mlperf style event.
   */
  void print_event_type(std::ostringstream& os, mlperf_logging::event_type et) const;

  /** @brief Populate log with value.
   *  @param ostringstream os Stores log string.
   *  @param event_type et Mlperf log value.
   */

  static size_t get_ms_since_epoch();

private:

  std::string m_sub_benchmark;
  std::string m_sub_org;
  std::string m_sub_division;
  std::string m_sub_status;
  std::string m_sub_platform;
  /* @brief name of output file. Default = results.txt */
  //std::string m_output_filename;
  /* @brief DiHydrogen logger */
  mutable h2::Logger m_logger{"mlperf_logger", "stdout", ":::MLLOG"};


}; // class mlperf_logging

std::unique_ptr<callback_base>
build_mlperf_logging_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_MLPERF_LOGGING_HPP_INCLUDED
