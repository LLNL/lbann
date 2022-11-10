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
   *  @param output_filename Output filename (default = results.txt)
   */
  mlperf_logging(std::string output_filename)
    : callback_base(/*batch_interval=*/1),
      m_output_filename{output_filename.size() ?
                        std::move(output_filename) :
                        std::string("results.txt")}
  {}

  /** @brief Copy interface */
  mlperf_logging* copy() const override {
    return new mlperf_logging(*this);
  }

  /** @brief Return name of callback */
  std::string name() const override { return "mlperf_logging"; }

  /** @brief Push mlperf formatted log string to stream object.
   *  @param ostream os Stores log strings.
   *  @param event_type et Type of mlperf style event.
   *  @param string key Mlperf log key.
   *  @param T value Mlperf log value.
   *  @param char const* file Current file name.
   *  @param size_t line File line number.
   *  @param double epoch Current epoch number.
   */
  template <typename T>
  void print(std::ostream& os, mlperf_logging::event_type et, std::string key,
             T value, char const* file, size_t line, double epoch = -1) const;

  void setup(model *m) override;
  void on_setup_end(model *m) override;
  void on_epoch_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_train_begin(model *m) override;
  void on_train_end(model *m) override;
  void on_batch_evaluate_begin(model *m) override;
  void on_batch_evaluate_end(model *m) override;

private:

  /** @brief Populate log with mlperf event type.
   *  @param ostream os Stores log string.
   *  @param event_type et Type of mlperf style event.
   */
  void print_event_type(std::ostream& os, mlperf_logging::event_type et) const;

  /** @brief Populate log with value.
   *  @param ostream os Stores log string.
   *  @param event_type et Mlperf log value.
   */
  void print_value(std::ostream& os, double value) const;
  void print_value(std::ostream& os, long value) const;
  void print_value(std::ostream& os, size_t value) const;
  void print_value(std::ostream& os, std::string value) const;
  //FIXME: Always picks this function first
  //template <typename T>
  //void print_value(std::ostream& os, T value) const;

  static size_t get_ms_since_epoch();

private:

  //FIXME: get logger to output file
  /* @brief name of output file. Default = results.txt */
  std::string m_output_filename;

  //FIXME: Add custom logging tag
  /* @brief DiHydrogen logger */
  h2::Logger m_logger;

}; // class mlperf_logging

std::unique_ptr<callback_base>
build_mlperf_logging_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_MLPERF_LOGGING_HPP_INCLUDED
