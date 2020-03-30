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
// freeze_layer .hpp .cpp - Callback hooks to time training
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_FREEZER_HPP_INCLUDED
#define LBANN_CALLBACKS_FREEZER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <map>

namespace lbann {
namespace callback {

/**
 */
class freezer : public callback_base {
 public:
  /// Type of the pair of epoch/step and layer name 
  using freeze_t = std::multimap<size_t, std::string>;

  freezer() = default;
  freezer(freeze_t&& freeze_e, freeze_t&& unfreeze_e,
          freeze_t&& freeze_s, freeze_t&& unfreeze_s);
  freezer(const freezer&) = default;
  freezer& operator=(const freezer&) = default;
  freezer* copy() const override {
    return new freezer(*this);
  }
  void on_epoch_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_batch_begin(model *m) override;
  void on_batch_end(model *m) override;
  std::string name() const override { return "freezer"; }

 private:
  freeze_t m_freeze_by_epoch;
  freeze_t m_unfreeze_by_epoch;
  freeze_t m_freeze_by_step;
  freeze_t m_unfreeze_by_step;
};

// Builder function
std::unique_ptr<callback_base>
build_freezer_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_FREEZER_HPP_INCLUDED
