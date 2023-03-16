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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_ALTERNATE_UPDATES_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_ALTERNATE_UPDATES_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * Alternate layer/weight updates to support training networks like GANs.
 * Takes lists of layers to freeze and unfreeze in an alternating fashion.
 * Supports a separate number of updates for each set of layers between
 * freezing and unfreezing.
 */
class alternate_updates : public callback_base
{
public:
  alternate_updates(std::vector<std::string> layers_1,
                    std::vector<std::string> layers_2,
                    int iters_1 = 1,
                    int iters_2 = 1)
    : callback_base(1),
      m_layer_names_1(std::move(layers_1)),
      m_layer_names_2(std::move(layers_2)),
      m_iters_1(iters_1),
      m_iters_tot(iters_1 + iters_2)
  {}

  alternate_updates(const alternate_updates&) = default;
  alternate_updates& operator=(const alternate_updates&) = default;
  alternate_updates* copy() const override
  {
    return new alternate_updates(*this);
  }
  void setup(model* m) override;
  void on_batch_begin(model* m) override;
  std::string name() const override { return "alternate updates"; }

private:
  void write_specific_proto(lbann_data::Callback& proto) const final;
  std::vector<std::string> m_layer_names_1, m_layer_names_2;
  std::vector<Layer*> freeze_layers, unfreeze_layers;
  int m_iters_1, m_iters_tot;
};

// Builder function
std::unique_ptr<callback_base> build_alternate_updates_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_ALTERNATE_UPDATES_HPP_INCLUDED
