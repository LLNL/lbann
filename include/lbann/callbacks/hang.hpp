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
// hang .hpp .cpp - Callback to hang LBANN for debuggers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_HANG_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_HANG_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * Hang LBANN as training starts so debuggers can attach.
 * This will cause either a specific rank (in COMM_WORLD) or every rank to hang.
 * Attach to the hung ranks and set the hang flag to false with a debugger to
 * proceed.
 */
class hang : public callback_base {
 public:
  /**
   * @param rank_to_hang The rank to hang; -1 for every rank (default).
   */
  hang(int rank_to_hang = -1) :
    m_rank_to_hang(rank_to_hang) {}
  hang(const hang&) = default;
  hang& operator=(const hang&) = default;
  hang* copy() const override { return new hang(*this); }

  void setup(model* m) override;

  /// Hang on train begin.
  void on_train_begin(model* m) override;//  {
  //   if (m_rank_to_hang == -1 ||
  //       m_rank_to_hang == m->get_comm()->get_rank_in_world()) {
  //     // Set this flag to false with your debugger to resume execution.
  //     volatile bool lbann_hang = true;
  //     while (lbann_hang) {}
  //   }
  // }
  std::string name() const override { return "hang"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

 private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /// The rank that will hang; -1 for every rank.
  int m_rank_to_hang;
};

// Builder function
std::unique_ptr<callback_base>
build_hang_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_HANG_HPP_INCLUDED
