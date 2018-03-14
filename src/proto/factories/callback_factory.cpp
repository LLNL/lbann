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

#include "lbann/proto/factories.hpp"

namespace {

/** Select entries from a list based on names.
 *  Any entry in 'list' with a name found in 'names' (interpreted as a
 *  space-separated list) is added to the output set.
 */  
template <typename T>
std::unordered_set<T*> select_from_list(std::string names,
                                        std::vector<T*> list) {
  std::unordered_set<T*> selected;
  for (const auto& name : lbann::proto::parse_list<>(names)) {
    for (auto&& t : list) {
      if (name == t->get_name()) {
        selected.insert(t);
      }
    }
  }
  return selected;
}

} // namespace

namespace lbann {
namespace proto {

lbann_callback* construct_callback(lbann_comm* comm,
                                   const lbann_data::Callback& proto_cb,
                                   std::map<execution_mode, generic_data_reader*>& data_readers,
                                   std::vector<Layer*> layer_list,
                                   std::vector<weights*> weights_list,
                                   lbann_summary* summarizer) {
  std::stringstream err;

  //////////////////////////////////////////////////////////////////
  // Display information
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_print()) {
    const auto& interval = proto_cb.print().interval();
    return new lbann_callback_print(interval);
  }
  if (proto_cb.has_timer()) {
    return new lbann_callback_timer(summarizer);
  }
  if (proto_cb.has_disp_io_stats()) {
    const auto& proto_disp_io_stats = proto_cb.disp_io_stats();
    auto&& selected_layers = select_from_list<Layer>(proto_disp_io_stats.layers(),
                                                     layer_list);
    return new lbann_callback_io(selected_layers);
  }
  if (proto_cb.has_save_images()) {
    const auto& proto_save_images = proto_cb.save_images();
    const auto& image_dir = proto_save_images.image_dir();
    const auto& extension = proto_save_images.extension();
    auto&& reader = data_readers[execution_mode::training] ;
    return new lbann_callback_save_images(reader, image_dir, extension);
  }

  //////////////////////////////////////////////////////////////////
  // Inter-model communication
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_ltfb()) {
    const auto& round_size = proto_cb.ltfb().round_size();
    return new lbann_callback_ltfb(round_size, summarizer);
  }  
  /// @todo
  if (proto_cb.has_imcomm()) {
    const auto& proto_imcomm = proto_cb.imcomm();
    const auto& type_str = proto_imcomm.intermodel_comm_method();
    lbann_callback_imcomm::comm_type type = lbann_callback_imcomm::comm_type::NONE;
    if (type_str == "none") {
      type = lbann_callback_imcomm::comm_type::NONE;
    } else if (type_str == "normal") {
      type = lbann_callback_imcomm::comm_type::NORMAL;
    } else if (type_str == "onebit_quantization") {
      type = lbann_callback_imcomm::comm_type::ONEBIT_QUANTIZATION;
    } else if (type_str == "thresh_quantization") {
      type = lbann_callback_imcomm::comm_type::THRESH_QUANTIZATION;
    } else if (type_str == "adaptive_quantization") {
      type = lbann_callback_imcomm::comm_type::ADAPTIVE_QUANTIZATION;
    } else {
      err << "invalid inter-model communication type (" << type_str << ")";
      LBANN_ERROR(comm, err.str());
    }
    std::unordered_set<weights*> selected_weights; /// @todo Initialize weights
    return new lbann_callback_imcomm(type, selected_weights, summarizer);
  }

  //////////////////////////////////////////////////////////////////
  // Learning rate schedules
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_step_learning_rate()) {
    const auto& proto_step_lr = proto_cb.step_learning_rate();
    const auto& step = proto_step_lr.step();
    const auto& amt = proto_step_lr.amt();
    auto&& selected_weights = select_from_list<weights>(proto_step_lr.weights(),
                                                        weights_list);
    return new lbann_callback_step_learning_rate(step, amt, selected_weights);
  }
  if (proto_cb.has_adaptive_learning_rate()) {
    const auto& proto_adaptive_lr = proto_cb.adaptive_learning_rate();
    const auto& patience = proto_adaptive_lr.patience();
    const auto& amt = proto_adaptive_lr.amt();
    auto&& selected_weights = select_from_list<weights>(proto_adaptive_lr.weights(),
                                                        weights_list);
    return new lbann_callback_adaptive_learning_rate(patience, amt, selected_weights);
  }
  if (proto_cb.has_drop_fixed_learning_rate()) {
    const auto& proto_drop_fixed_lr = proto_cb.drop_fixed_learning_rate();
    std::vector<int64_t> drop_epochs;
    for (int i = 0; i < proto_drop_fixed_lr.drop_epoch_size(); ++i) {
      drop_epochs.push_back(proto_drop_fixed_lr.drop_epoch(i));
    }
    const auto& amt = proto_drop_fixed_lr.amt();
    auto&& selected_weights = select_from_list<weights>(proto_drop_fixed_lr.weights(),
                                                        weights_list);
    return new lbann_callback_drop_fixed_learning_rate(drop_epochs, amt, selected_weights);
  }
  if (proto_cb.has_linear_growth_learning_rate()) {
    const auto& proto_linear_growth_lr = proto_cb.linear_growth_learning_rate();
    const auto& target = proto_linear_growth_lr.target();
    const auto& num_epochs = proto_linear_growth_lr.num_epochs();
    const auto& delay = proto_linear_growth_lr.delay();
    auto&& selected_weights = select_from_list<weights>(proto_linear_growth_lr.weights(),
                                                        weights_list);
    return new lbann_callback_linear_growth_learning_rate(target, num_epochs, delay, selected_weights);
  }
  if (proto_cb.has_optimizerwise_adaptive_learning_rate()) {
    const auto& proto_adaptive_lr = proto_cb.optimizerwise_adaptive_learning_rate();
    const auto& scale = proto_adaptive_lr.scale();
    auto&& selected_weights = select_from_list<weights>(proto_adaptive_lr.weights(),
                                                        weights_list);
    return new lbann_callback_optimizerwise_adaptive_learning_rate(scale, selected_weights);
  }

  //////////////////////////////////////////////////////////////////
  // Mini-batch schedules
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_step_minibatch()) {
    const auto& proto_step_minibatch = proto_cb.step_minibatch();
    const auto& starting_mbsize = proto_step_minibatch.starting_mbsize();
    const auto& step = proto_step_minibatch.step();
    const auto& ramp_time = proto_step_minibatch.ramp_time();
    return new lbann_callback_step_minibatch(starting_mbsize, step, ramp_time);
  }
  if (proto_cb.has_minibatch_schedule()) {
    const auto& proto_minibatch_schedule = proto_cb.minibatch_schedule();
    const auto& starting_mbsize = proto_minibatch_schedule.starting_mbsize();
    std::vector<lbann_callback_minibatch_schedule::minibatch_step> steps;
    for (int i = 0; i < proto_minibatch_schedule.step_size(); ++i) {
      const auto& proto_step = proto_minibatch_schedule.step(i);
      const auto& epoch = proto_step.epoch();
      const auto& mbsize = proto_step.mbsize();
      const auto& lr = proto_step.lr();
      const auto& ramp_time = proto_step.ramp_time();
      steps.emplace_back(epoch, mbsize, lr, ramp_time);
    }
    return new lbann_callback_minibatch_schedule(starting_mbsize, steps);
  }

  //////////////////////////////////////////////////////////////////
  // Checkpointing and exporting
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_checkpoint()) {
    const auto& proto_checkpoint = proto_cb.checkpoint();
    const auto& dir = proto_checkpoint.checkpoint_dir();
    const auto& epochs = proto_checkpoint.checkpoint_epochs();
    const auto& steps = proto_checkpoint.checkpoint_steps();
    const auto& secs = proto_checkpoint.checkpoint_secs();
    const auto& per_rank = proto_checkpoint.checkpoint_per_rank();
    return new lbann_callback_checkpoint(dir, epochs, steps, secs, per_rank);
  }
  if (proto_cb.has_save_model()) {
    const auto& proto_save_model = proto_cb.save_model();
    const auto& dir = proto_save_model.dir();
    const auto& extension = proto_save_model.extension();
    return new lbann_callback_save_model(dir, extension);
  }

  //////////////////////////////////////////////////////////////////
  // Profiling
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_summary()) {
    const auto& proto_summary = proto_cb.summary();
    const auto& batch_interval = proto_summary.batch_interval();
    const auto& mat_interval = proto_summary.mat_interval();
    return new lbann_callback_summary(summarizer, batch_interval, mat_interval);
  }
  if (proto_cb.has_profiler()) {
    return new lbann_callback_profiler();
  }

  //////////////////////////////////////////////////////////////////
  // Debugging
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_debug()) {
    const auto& phase = proto_cb.debug().phase();
    if (phase == "train" || phase == "training") {
      return new lbann_callback_debug(execution_mode::training, summarizer);
    } else if (phase == "validate" || phase == "validation") {
      return new lbann_callback_debug(execution_mode::validation, summarizer);
    } else if (phase == "test" || phase == "testing") {
      return new lbann_callback_debug(execution_mode::testing, summarizer);
    } else {
      return new lbann_callback_debug();
    }
  }
  if (proto_cb.has_debug_io()) {
    const auto& proto_debug_io = proto_cb.debug_io();
    const auto& phase = proto_debug_io.phase();
    const auto& lvl = proto_debug_io.lvl();
    if (phase == "train" || phase == "training") {
      return new lbann_callback_debug_io(execution_mode::training, lvl);
    } else if (phase == "validate" || phase == "validation") {
      return new lbann_callback_debug_io(execution_mode::validation, lvl);
    } else if (phase == "test" || phase == "testing") {
      return new lbann_callback_debug_io(execution_mode::testing, lvl);
    } else {
      return new lbann_callback_debug_io();
    }
  }
  if (proto_cb.has_dump_weights()) {
    const auto& proto_dump_weights = proto_cb.dump_weights();
    const auto& basename = proto_dump_weights.basename();
    const auto& interval = proto_dump_weights.interval();
    return new lbann_callback_dump_weights(basename, interval);
  }
  if (proto_cb.has_dump_activations()) {
    const auto& proto_dump_acts = proto_cb.dump_activations();
    const auto& basename = proto_dump_acts.basename();
    const auto& interval = proto_dump_acts.interval();
    const auto& layer_names = parse_list<>(proto_dump_acts.layer_names());
    return new lbann_callback_dump_activations(basename, interval, layer_names);
  }
  if (proto_cb.has_dump_gradients()) {
    const auto& proto_dump_gradients = proto_cb.dump_gradients();
    const auto& basename = proto_dump_gradients.basename();
    const auto& interval = proto_dump_gradients.interval();
    return new lbann_callback_dump_gradients(basename, interval);
  }
  if (proto_cb.has_dump_mb_indices()) {
    const auto& proto_dump_mb_indices = proto_cb.dump_mb_indices();
    const auto& basename = proto_dump_mb_indices.basename();
    const auto& interval = proto_dump_mb_indices.interval();
    return new lbann_callback_dump_minibatch_sample_indices(basename, interval);
  }
  if (proto_cb.has_check_dataset()) {
    return new lbann_callback_check_dataset();
  }
  if (proto_cb.has_check_small()) {
    return new lbann_callback_checksmall();
  }
  if (proto_cb.has_check_nan()) {
    return new lbann_callback_checknan();
  }
  if (proto_cb.has_hang()) {
    const auto& rank_to_hang = proto_cb.hang().rank();
    if (comm->am_world_master()) {
      if (rank_to_hang == -1) {
        std::cout << "*** HANGING EVERY RANK IN HANG CALLBACK ***"
                  << std::endl;
      } else {
        std::cout << "*** HANGING RANK " << rank_to_hang
                  << " IN HANG CALLBACK ***" << std::endl;
      }
    }
    return new lbann_callback_hang(rank_to_hang);
  }

  //////////////////////////////////////////////////////////////////
  // Gradient checking
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_gradient_check()) {
    const auto& proto_gradient_check = proto_cb.gradient_check();
    const auto& step_size = proto_gradient_check.step_size();
    const auto& verbose = proto_gradient_check.verbose();
    const auto& fail_on_error = proto_gradient_check.fail_on_error();
    return new lbann_callback_gradient_check(step_size, verbose, fail_on_error);
  }

  return nullptr;

}

} // namespace proto
} // namespace lbann
