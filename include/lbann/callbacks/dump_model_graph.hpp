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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_MODEL_GRAPH_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_MODEL_GRAPH_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * @brief Dump model graph callback.
 *
 * This callback dumps a graphviz graph that represents the model at
 * the end of setup.
 */
class dump_model_graph : public callback_base {
 public:
  dump_model_graph(std::string basename, bool print) :
      m_basename(basename), m_print(print) {}
  dump_model_graph(const dump_model_graph&) = default;
  dump_model_graph& operator=(
    const dump_model_graph&) = default;
  dump_model_graph* copy() const override {
    return new dump_model_graph(*this);
  }
  std::string name() const override { return "print tensor dimensions"; }

  void on_setup_end(model *m) override;

 private:
  /** Filename to output graphviz graph. */
  std::string m_basename;
  /** Whether to print the model architecture to stdout. */
  bool m_print;

};

// Builder function
std::unique_ptr<callback_base>
build_dump_model_graph_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_MODEL_GRAPH_HPP_INCLUDED
