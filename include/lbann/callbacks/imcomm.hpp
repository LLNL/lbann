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
// imcomm .hpp .cpp - Send gradient updates between models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lbann {

template <typename T>
class data_type_weights;

namespace callback {

/**
 * @brief Support inter-model communication after each mini-batch to
 *        synchronize gradient updates.
 */
class imcomm : public callback_base
{
public:
  using callback_base::on_backward_prop_end;

  enum comm_type
  {
    NONE = 0, /** Do no gradient updates. */
    NORMAL,   /** Simply sum gradient updates. */
  };

  /**
   * @brief Initialize with ct being used for all weights.
   */
  imcomm(comm_type ct = NORMAL,
         const std::shared_ptr<lbann_summary>& summarizer = nullptr);
  imcomm(const imcomm&) = default;
  imcomm& operator=(const imcomm&) = default;
  imcomm* copy() const override { return new imcomm(*this); }
  /**
   * @brief Convenience initialization to do one update type for specific
   * weights.
   *
   * @details Implies no inter-model updates for other weights.
   */
  imcomm(comm_type ct,
         std::unordered_set<weights*> weights_list,
         const std::shared_ptr<lbann_summary>& summarizer = nullptr);

  /** @brief Choose comm type ct for weights. */
  void set_weights_comm(weights* w, comm_type ct);

  /** @brief Do initialization for this model. */
  void setup(model* m) override;

  /** @brief Make sure all models have the same weights. */
  void on_train_begin(model* m) override;

  /** @brief Do inter-model gradient updates. */
  void on_backward_prop_end(model* m) override;

  std::string name() const override { return "imcomm"; }

private:
  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

  /** @brief Summarize relevant statistics. */
  template <typename T>
  void do_summary(model const& m, data_type_weights<T>& w, EvalType im_time);

private:
  /** @brief Parameters for a given set of weights. */
  struct imcomm_params
  {
    /** @brief Type of communication done. */
    comm_type ct = NONE;
  };

  /** @brief Default communication type. */
  comm_type m_default_ct;

  /** @brief Per-weights parameters. */
  std::unordered_map<weights*, imcomm_params> m_weights_params;

  /** @brief @brief lbann_summary */
  std::shared_ptr<lbann_summary> m_summarizer = nullptr;
};

/** @brief returns a string representation of the weight_initialization */
std::string get_comm_type_name(typename imcomm::comm_type m);

// Builder function
std::unique_ptr<callback_base>
build_imcomm_callback_from_pbuf(const google::protobuf::Message&,
                                std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
