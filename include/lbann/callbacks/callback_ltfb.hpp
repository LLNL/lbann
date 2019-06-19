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

#ifndef LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <memory>
#include <set>
#include <vector>

namespace lbann {

/** @brief Tournament training.
 *
 *  This is intended to support research into the LTFB algorithm. An
 *  outline:
 *    - Divide the computational resources into multiple "trainers"
 *      that can operate in parallel.
 *    - Setup a model on each trainer and begin training independently.
 *    - Periodically launch tournaments to select "good" models. More
 *      specifically, trainers partner up and exchange their models.
 *      Each trainer evaluates a metric for its local and partner
 *      models, using its validation data set. The model with the better
 *      score is retained and the other one is discarded.
 *
 *  There are many algorithmic variations to be explored:
 *    - How is data is divvied up amongst the trainers. Is it strictly
 *      partitioned, partially shared, or completely replicated?
 *    - What model components are exchanged? Just the trainable weights,
 *      or a subset of the weights? Hyperparameters?
 *    - Can this be used to explore model architectures?
 *
 *  @todo Exchange optimizer state.
 *  @todo Support heterogeneous models.
 */
class lbann_callback_ltfb : public lbann_callback {
public:

  /** Inter-trainer communication scheme for LTFB.
   *
   *  The specifics of these algorithms are experimental and will be
   *  in flux.
   */
  enum class communication_algorithm {
    /** Directly exchange weights values with sendrecv.
     *
     *  Corresponding ranks in partner trainers will iterate through
     *  their weights and exchange values with sendrecvs.
     *
     *  Notes:
     *    - Requires all models to be identical aside from their
     *      weights values, so this is not suitable for hyperparameter
     *      or model architecture exploration.
     *    - Optimizer state is not exchanged, so there may be wonky
     *      learning behavior immediately after a tournament.
     *    - Optimal if communication performance between ranks is
     *      uniform and independent. If intra-trainer communication is
     *      fast or if communication performance is sensitive to
     *      network traffic, it may be advantageous to gather model
     *      data on the trainer master ranks and only perform
     *      inter-trainer communication between them.
     */
    sendrecv_weights,

    /** Save and load model data with checkpoint files.
     *
     *  @todo Implement.
     *
     *  Notes:
     *    - Supports hyperparameter exploration.
     *    - Checkpoint files currently do not store model architecture
     *      information, so this is not suitable for model
     *      architecture exploraiton.
     *    - This approach is temporary and experimental, since going
     *      through the file system is very suboptimal. When a wire
     *      format for model checkpoints is developed, it should be
     *      used instead.
     */
    checkpoint_file
  };

  /** @brief Construct the LTFB callback
   *  @param batch_interval Number of training mini-batch steps between
   *                        tournaments.
   *  @param metric_name    Metric for tournament evaluation.
   *  @param weights_names  List of weights to exchange with partner.
   *                        If empty, then all weights are exchanged.
   *  @param low_score_wins Whether low-scoring or high-scoring models
   *                        survive a tournament.
   *  @param comm_algo      Inter-trainer communication scheme.
   *  @param summarizer     The summarizer to use for this callback
   */
  lbann_callback_ltfb(
    El::Int batch_interval,
    std::string metric_name,
    std::set<std::string> weights_names = std::set<std::string>(),
    bool low_score_wins = false,
    communication_algorithm comm_algo = communication_algorithm::sendrecv_weights,
    bool exchange_hyperparameters = false,
    lbann_summary *summarizer = nullptr);
  lbann_callback_ltfb(const lbann_callback_ltfb& other);
  lbann_callback_ltfb& operator=(const lbann_callback_ltfb& other);
  lbann_callback_ltfb* copy() const override { return new lbann_callback_ltfb(*this); }
  std::string name() const override { return "LTFB"; }

  void setup(model *m) override;
  void on_train_begin(model *m) override;
  void on_batch_begin(model *m) override;

  /** Convert string to LTFB communication algorithm.
   *
   *  If an empty string is provided, returns @c
   *  communication_algorithm::sendrecv_weights.
   */
  static communication_algorithm string_to_comm_algo(const std::string& str);

private:

  /** Metric for tournament evaluation. */
  std::string m_metric_name;

  /** List of weights to exchange with partner.
   *
   *  If empty, then all weights are exchanged.
   */
  std::set<std::string> m_weights_names;

  /** Whether low-scoring or high-scoring models survive a
   *  tournament. */
  bool m_low_score_wins;

  /** Inter-trainer communication scheme. */
  communication_algorithm m_comm_algo;
 
  /** Whether to exchange training hyperparameters between trainers
  */
  bool m_exchange_hyperparameters;

  /** Workspace weights.
   *
   *  Used to temporarily store local weights during a tournament.
   */
  std::vector<std::unique_ptr<weights>> m_workspace_weights;

};

} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
