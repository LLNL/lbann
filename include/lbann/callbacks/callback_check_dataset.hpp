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

#ifndef LBANN_CALLBACKS_CALLBACK_CHECK_DATASET_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECK_DATASET_HPP_INCLUDED

#include <set>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 * Save the sample indices for each mini-batch to ordered set.
 * Check to make sure that all samples were properly processed.
 */
class lbann_callback_check_dataset : public lbann_callback {
 public:
  using lbann_callback::on_forward_prop_end;
  using lbann_callback::on_evaluate_forward_prop_end;

  lbann_callback_check_dataset() :
    lbann_callback() {}
  lbann_callback_check_dataset(
    const lbann_callback_check_dataset&) = default;
  lbann_callback_check_dataset& operator=(
    const lbann_callback_check_dataset&) = default;
  lbann_callback_check_dataset* copy() const override {
    return new lbann_callback_check_dataset(*this);
  }
  void on_forward_prop_end(model *m, Layer *l) override;
  void on_evaluate_forward_prop_end(model *m, Layer *l) override;
  void on_epoch_end(model *m) override;
  void on_validation_end(model *m) override;
  void on_test_end(model *m) override;

  void add_to_set(model *m, Layer *l, int64_t step, std::set<long> &set);

  std::string name() const override { return "check data set indices"; }
 private:
  /** @brief Basename for writing files. */
  std::string m_basename;

  std::set<long> training_set;
  std::set<long> validation_set;
  std::set<long> testing_set;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECK_DATASET_HPP_INCLUDED
