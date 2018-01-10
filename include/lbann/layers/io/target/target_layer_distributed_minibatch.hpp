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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/data_distributions/distributed_minibatch.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <data_layout T_layout>
class target_layer_distributed_minibatch : public target_layer, public distributed_minibatch {
 protected:
  Mat Y_local; /** Local matrix that holds data from data reader */
  Mat Y_local_v; /** View of local matrix that holds data from data reader */
  CircMat Ys; /** Distributed matrix used to stage local data to layer output */

 public:
  target_layer_distributed_minibatch(lbann_comm *comm, input_layer *input_layer, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, bool shared_data_reader, bool for_regression = false)
    : generic_data_distribution(comm, num_parallel_readers, data_readers),
      target_layer(comm, input_layer, data_readers, for_regression),
      distributed_minibatch(comm, num_parallel_readers, data_readers),
      Ys(comm->get_model_grid()) {

    // if(!dynamic_cast<input_layer_distributed_minibatch*>(input_layer)) {
    //   std::stringstream err;
    //   err << __FILE__ << " " << __LINE__ 
    //       << " :: " << get_type() << " paired with invalid input layer type" << std::endl;
    //   throw lbann_exception(err.str());
    // }
    generic_data_distribution::fetch_data_fn = new fetch_data_functor(false, target_layer::is_for_regression());
    generic_data_distribution::update_data_reader_fn = new update_data_reader_functor(false);
  }
  target_layer_distributed_minibatch(
    const target_layer_distributed_minibatch&) = default;
  target_layer_distributed_minibatch& operator=(
    const target_layer_distributed_minibatch&) = default;
  target_layer_distributed_minibatch* copy() const override {
    return new target_layer_distributed_minibatch(*this);
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} + " target_layer_distributed_minibatch "
           + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  std::string get_type() const override { return "target:distributed"; }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_data() override {
    target_layer::setup_data();

    int max_mb_size = this->m_model->get_max_mini_batch_size();
    Y_local.Resize(this->m_num_neurons, max_mb_size);
    Ys.Resize(this->m_num_neurons, max_mb_size);

    m_local_data_valid = false;
    m_local_reader_done = false;
    m_num_data_per_epoch = 0;
  }

  void fp_setup_data() override {
    target_layer::fp_setup_data();
    El::Int cur_mini_batch_size = m_model->get_current_mini_batch_size();
    El::View(Y_local_v, Y_local, El::ALL, El::IR(0, cur_mini_batch_size));
  }

  void fp_compute() override {

    int num_samples_in_batch = fetch_to_local_matrix(Y_local_v, paired_input_layer->get_data_reader());
    if(is_current_root()) {
      /// Only update the number of samples processed by this parallel reader, when it is the current root
      target_layer::update_num_samples_processed(num_samples_in_batch);
    }

    int curr_mini_batch_size = this->m_model->get_current_mini_batch_size();
    if(is_current_root() && num_samples_in_batch != curr_mini_batch_size) {
      throw lbann_exception("lbann_target_layer_distributed_minibatch: number of labels ("
                            + std::to_string(num_samples_in_batch) + ") does not match the current mini-batch size (" 
                            + std::to_string(curr_mini_batch_size) + ")."
                            );
    }
    /// @todo should this distribute the entire matrix even if there is only a partial mini-batch
    distribute_from_local_matrix(Y_local, Ys, paired_input_layer->get_data_reader());

    const auto& prediction = get_prediction();
    m_ground_truth->Resize(prediction.Height(), prediction.Width());
    Copy(Ys, *m_ground_truth);

    return;
  }

  void bp_compute() override {}

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() override {
    return is_data_set_processed(paired_input_layer->get_data_reader());
  }

  void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) override {
    return;
  }

};

}  // namespace lbann

#endif  // LBANN_LAYERS_TARGET_LAYER_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
