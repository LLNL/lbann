////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_METRIC_EXECUTABLE_METRIC_HPP
#define LBANN_METRIC_EXECUTABLE_METRIC_HPP

#include "lbann/metrics/metric.hpp"

namespace lbann {

/** @brief A metric that receives its value from parsing the output of a running
 *  executable.
 *
 *  This metric spawns an executable file with every evaluation and reads its
 *  output to receive one value of type ``EvalType``. It expects the program
 *  to have no other output to ``stdout``, and will be called with the following
 *  command-line arguments: ``<filename> [other_args] <experiment directory>``.
 */
class executable_metric : public metric
{

public:
  executable_metric(lbann_comm* comm = nullptr,
                    std::string name = "",
                    std::string filename = "",
                    std::string other_args = "")
    : metric(comm), m_name(name), m_filename(filename), m_other_args(other_args)
  {}
  executable_metric(const executable_metric& other) = default;
  executable_metric& operator=(const executable_metric& other) = default;
  virtual ~executable_metric() = default;
  executable_metric* copy() const override
  {
    return new executable_metric(*this);
  }

  /** Return a string name for this metric. */
  std::string name() const override;

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** Get list of pointers to layers. */
  std::vector<ViewingLayerPtr> get_layer_pointers() const override;
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<ViewingLayerPtr> layers) override;

  /** Save metric state to checkpoint. */
  bool save_to_checkpoint_shared(persist& p) override;
  /** Load metric state from checkpoint. */
  bool load_from_checkpoint_shared(persist& p) override;

  bool save_to_checkpoint_distributed(persist& p) override;
  bool load_from_checkpoint_distributed(persist& p) override;

protected:
  void setup(model& m) override;
  EvalType evaluate(execution_mode mode, int mini_batch_size) override;

private:
  /** Descriptive name for metric. */
  std::string m_name;

  /** Path to executable to run. */
  std::string m_filename;

  /** Arguments to prepend before experiment path. */
  std::string m_other_args;
};

} // namespace lbann

#endif // LBANN_METRIC_EXECUTABLE_METRIC_HPP
