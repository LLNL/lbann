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
// dump_minibatch_sample_indices .hpp .cpp - Callbacks
// to dump the list of indices per minibatch
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_MINIBATCH_SAMPLE_INDICES_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_MINIBATCH_SAMPLE_INDICES_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * @brief Dump sample indices for each minibatch to files.
 * @details This will dump the list of indices from the training /
 * validation / testing data that was processed Note this dumps
 * vectors during each mini-batch. This will be slow and produce a lot
 * of output.
 */
class dump_minibatch_sample_indices : public callback_base
{
public:
  using callback_base::on_evaluate_forward_prop_end;
  using callback_base::on_forward_prop_end;

  /**
   * @param basename The basename for writing files.
   * @param batch_interval The frequency at which to dump sample indices
   */
  dump_minibatch_sample_indices(std::string basename, int batch_interval = 1)
    : callback_base(batch_interval), m_basename(std::move(basename))
  {}
  dump_minibatch_sample_indices(const dump_minibatch_sample_indices&) = default;
  dump_minibatch_sample_indices&
  operator=(const dump_minibatch_sample_indices&) = default;
  dump_minibatch_sample_indices* copy() const override
  {
    return new dump_minibatch_sample_indices(*this);
  }
  void on_forward_prop_end(model* m, Layer* l) override;
  void on_evaluate_forward_prop_end(model* m, Layer* l) override;

  void dump_to_file(model* m, Layer* l, int64_t step);

  std::string name() const override { return "dump minibatch sample indices"; }

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
  dump_minibatch_sample_indices();

  /** Basename for writing files. */
  std::string m_basename;
};

// Builder function
std::unique_ptr<callback_base>
build_dump_mb_indices_callback_from_pbuf(const google::protobuf::Message&,
                                         std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_DUMP_MINIBATCH_SAMPLE_INDICES_HPP_INCLUDED
