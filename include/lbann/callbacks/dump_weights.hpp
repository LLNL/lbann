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
// dump_weights .hpp .cpp - Callbacks to dump weight matrices
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_WEIGHTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_WEIGHTS_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"
#include "lbann/utils/visitor_hooks.hpp"

namespace lbann {
namespace callback {

// Forward declaration
namespace dump_weights_internal {
class FileFormat;
}

/** @brief Dump weights to files.
 *
 *  Saves all weights to files. This is meant to export the weights
 *  for debugging and analysis that isn't easily done in LBANN. It is
 *  not meant for checkpointing.
 *
 *  The "text" and "binary" formats are written using Elemental's
 *  ASCII and BINARY formats, respectively. The "distributed_binary"
 *  format is written by using Elemental's BINARY format independently
 *  on each process' local data.
 */
class dump_weights : public callback_base {
 public:
   /** @brief Constructor
    *  @param dir Directory in which weight files will be saved.
    *  @param epoch_interval The number of epochs between weights.
    *  @param file_format The output file format.
   */
  dump_weights(
    std::string dir,
    El::Int epoch_interval,
    std::unique_ptr<dump_weights_internal::FileFormat> file_format);
  dump_weights(const dump_weights&);
  dump_weights& operator=(const dump_weights&);
  dump_weights* copy() const override {
    return new dump_weights(*this);
  }
  void on_train_begin(model *m) override;
  void on_epoch_end(model *m) override;
  std::string name() const override { return "dump weights"; }
  void set_target_dir(const std::string& dir) { m_directory = dir; }
  const std::string& get_target_dir() { return m_directory; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

 private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  friend class cereal::access;
  dump_weights();

  /// Basename for writing files
  std::string m_directory;
  /// Interval at which to dump weights
  El::Int m_epoch_interval;
  /// Weight file format
  std::unique_ptr<dump_weights_internal::FileFormat> m_file_format;

  /// Dump weights from learning layers
  void do_dump_weights(const model& m, visitor_hook hook);

};

// Builder function
std::unique_ptr<callback_base>
build_dump_weights_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_DUMP_WEIGHTS_HPP_INCLUDED
