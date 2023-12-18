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
//
// progress_bar .hpp .cpp - Callback that prints a progress bar during epochs.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_PROGRESS_BAR_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_PROGRESS_BAR_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <array>

namespace lbann {
namespace callback {

/**
 * @brief prints a progress bar during training
 */
class progress_bar : public callback_base
{
public:
  using callback_base::on_epoch_begin;
  using callback_base::on_forward_prop_begin;

  /**
   * @param batch_interval The frequency at which to print the progress bar.
   * @param newline_interval The frequency at which to print a new line.
   * @param print_mem_usage If true, prints current GPU memory usage.
   * @param moving_average_length The number of iterations to compute a moving
   *                              average of iteration time on.
   * @param bar_width The width (in characters) of the printed progress bar.
   */
  progress_bar(int batch_interval = 1,
               int newline_interval = 0,
               bool print_mem_usage = false,
               int moving_average_length = 10,
               int bar_width = 30)
    : callback_base(batch_interval),
      m_newline_interval(newline_interval),
      m_print_mem_usage(print_mem_usage),
      m_moving_average_length(moving_average_length),
      m_bar_width(bar_width)
  {}
  progress_bar(const progress_bar&) = default;
  progress_bar& operator=(const progress_bar&) = default;
  progress_bar* copy() const override { return new progress_bar(*this); }

  void on_epoch_begin(model* m) override;
  void on_forward_prop_begin(model* m) override;

  std::string name() const override { return "progress bar"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  friend class cereal::access;

  // Settings
  int m_newline_interval;
  bool m_print_mem_usage;
  int m_moving_average_length;
  int m_bar_width;

  // Cached values for epochs
  bool m_print;
  int m_training_iterations;
  int m_current_iteration;

  /** Set up a rolling buffer for a moving average */
  std::vector<double> m_moving_avg_time;

  /** Time measurement (last time forward prop was called). */
  double m_last_time;
};

// Builder function
std::unique_ptr<callback_base>
build_progress_bar_callback_from_pbuf(const google::protobuf::Message&,
                                      std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_PROGRESS_BAR_HPP_INCLUDED
