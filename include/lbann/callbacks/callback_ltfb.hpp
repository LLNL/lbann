////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// lbann_callback_ltfb .hpp .cpp - Manage LTFB training for a model
////////////////////////////////////////////////////////////////////////////////

#ifndef __LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
#define __LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Manage LTFB training.
 * LTFB works in rounds, which are made up of some number of mini-batches (that
 * evenly divide the number of minibatches in an epoch). In each round, the
 * model trains as usual, and at the end it is randomly paired with another
 * model. The pairs exchange their models and evaluate both their local and the
 * received model on their validation data. The model achieving the highest
 * accuracy is retained and training continues.
 * Current limitations:
 * - Does not transfer optimizer state, so it's best to stick to SGD without
 * momentum.
 * - Uses the validation data for the tournament (we may not want this).
 * - Requires a manually-created model duplicate.
 */
class lbann_callback_ltfb : public lbann_callback {
 public:
  /**
   * Initialize LFTB.
   * @param round_size The number of minibatches in each round.
   * @param remote_model A duplicate of the model being trained (temp workaround).
   */
  lbann_callback_ltfb(uint round_size,
                      lbann_summary *summarizer = nullptr);
  lbann_callback_ltfb(const lbann_callback_ltfb& other);
  lbann_callback_ltfb& operator=(const lbann_callback_ltfb& other);
  ~lbann_callback_ltfb() override;
  lbann_callback_ltfb* copy() const override { return new lbann_callback_ltfb(*this); }
  /** Set up LTFB. */
  void setup(model *m) override;
  /**
   * Potentially run an LTFB round.
   */
  void on_batch_begin(model *m) override;

  std::string name() const override { return "ltfb"; }
 private:
  lbann_comm *m_comm;
  /** Number of minibatches in a round. */
  uint m_round_size;
  /** Second model for doing the tournament. */
  model *m_remote_model = nullptr;

  /**
   * Global operation that selects partners for the current tournament.
   * This generates unique pairs (i.e. each model competes with only one other
   * model). If there is an odd number of models, one of them sits out.
   * @return The local rank's partner.
   */
  int select_partner();
  /**
   * Exchange local model data with partner's.
   */
  void exchange(model *m, int partner);
  /**
   * Evaluate a model on tournament data and return its accuracy.
   */
  EvalType evaluate(model *m);
  /**
   * Replace the local model m with the remote model data.
   */
  void replace_with_remote(model *m);
};

}  // namespace lbann

#endif  // __LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
