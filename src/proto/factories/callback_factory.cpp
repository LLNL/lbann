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
#include "lbann/utils/peek_map.hpp"

namespace lbann {
namespace proto {

namespace {

/** Select entries from a list based on names.
 *  Any entry in 'list' with a name found in 'names' (interpreted as a
 *  space-separated list) is added to the output set.
 */  
template <typename T>
std::unordered_set<T*> select_from_list(std::string names,
                                        std::vector<T*> list) {
  std::unordered_set<T*> selected;
  for (const auto& name : parse_list<std::string>(names)) {
    for (auto&& t : list) {
      if (name == t->get_name()) {
        selected.insert(t);
      }
    }
  }
  return selected;
}

} // namespace

lbann_callback* construct_callback(lbann_comm* comm,
                                   const lbann_data::Callback& proto_cb,
                                   const std::map<execution_mode, generic_data_reader*>& data_readers,
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
    const auto& params = proto_cb.disp_io_stats();
    auto&& selected_layers = select_from_list<Layer>(params.layers(),
                                                     layer_list);
    return new lbann_callback_io(selected_layers);
  }
  if (proto_cb.has_save_images()) {
    const auto& params = proto_cb.save_images();
    auto&& reader = lbann::peek_map(data_readers, execution_mode::training);
    const auto& layer_names = parse_list<>(params.layer_names());
    return new lbann_callback_save_images(reader,
                                          params.image_dir(),
                                          layer_names,
                                          params.extension());
  }

  //////////////////////////////////////////////////////////////////
  // Inter-model communication
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_ltfb()) {
    return new lbann_callback_ltfb(proto_cb.ltfb().round_size(),
                                   summarizer);
  }  
  /// @todo
  if (proto_cb.has_imcomm()) {
    const auto& params = proto_cb.imcomm();
    const auto& type_str = params.intermodel_comm_method();
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
      LBANN_ERROR(err.str());
    }
    std::unordered_set<weights*> selected_weights; /// @todo Initialize weights
    return new lbann_callback_imcomm(type, selected_weights, summarizer);
  }

  //////////////////////////////////////////////////////////////////
  // Learning rate schedules
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_step_learning_rate()) {
    const auto& params = proto_cb.step_learning_rate();
    auto&& selected_weights = select_from_list<weights>(params.weights(),
                                                        weights_list);
    return new lbann_callback_step_learning_rate(params.step(),
                                                 params.amt(),
                                                 selected_weights);
  }
  if (proto_cb.has_adaptive_learning_rate()) {
    const auto& params = proto_cb.adaptive_learning_rate();
    auto&& selected_weights = select_from_list<weights>(params.weights(),
                                                        weights_list);
    return new lbann_callback_adaptive_learning_rate(params.patience(),
                                                     params.amt(),
                                                     selected_weights);
  }
  if (proto_cb.has_drop_fixed_learning_rate()) {
    const auto& params = proto_cb.drop_fixed_learning_rate();
    std::vector<int64_t> drop_epochs;
    for (int i = 0; i < params.drop_epoch_size(); ++i) {
      drop_epochs.push_back(params.drop_epoch(i));
    }
    auto&& selected_weights = select_from_list<weights>(params.weights(),
                                                        weights_list);
    return new lbann_callback_drop_fixed_learning_rate(drop_epochs,
                                                       params.amt(),
                                                       selected_weights);
  }
  if (proto_cb.has_linear_growth_learning_rate()) {
    const auto& params = proto_cb.linear_growth_learning_rate();
    auto&& selected_weights = select_from_list<weights>(params.weights(),
                                                        weights_list);
    return new lbann_callback_linear_growth_learning_rate(params.target(),
                                                          params.num_epochs(),
                                                          params.delay(),
                                                          selected_weights);
  }
  if (proto_cb.has_optimizerwise_adaptive_learning_rate()) {
    const auto& params = proto_cb.optimizerwise_adaptive_learning_rate();
    auto&& selected_weights = select_from_list<weights>(params.weights(),
                                                        weights_list);
    return new lbann_callback_optimizerwise_adaptive_learning_rate(params.scale(),
                                                                   selected_weights);
  }
  if (proto_cb.has_poly_learning_rate()) {
    const auto& params = proto_cb.poly_learning_rate();
    auto&& selected_weights = select_from_list<weights>(params.weights(),
                                                        weights_list);
    return new lbann_callback_poly_learning_rate(params.power(),
                                                 params.num_epochs(),
                                                 params.max_iter(),
                                                 selected_weights);
  }

  //////////////////////////////////////////////////////////////////
  // Mini-batch schedules
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_step_minibatch()) {
    const auto& params = proto_cb.step_minibatch();
    return new lbann_callback_step_minibatch(params.starting_mbsize(),
                                             params.step(),
                                             params.ramp_time());
  }
  if (proto_cb.has_minibatch_schedule()) {
    const auto& params = proto_cb.minibatch_schedule();
    std::vector<lbann_callback_minibatch_schedule::minibatch_step> steps;
    for (int i = 0; i < params.step_size(); ++i) {
      const auto& proto_step = params.step(i);
      steps.emplace_back(proto_step.epoch(),
                         proto_step.mbsize(),
                         proto_step.lr(),
                         proto_step.ramp_time());
    }
    return new lbann_callback_minibatch_schedule(params.starting_mbsize(),
                                                 steps);
  }

  //////////////////////////////////////////////////////////////////
  // Checkpointing and exporting
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_checkpoint()) {
    const auto& params = proto_cb.checkpoint();
    return new lbann_callback_checkpoint(params.checkpoint_dir(),
                                         params.checkpoint_epochs(),
                                         params.checkpoint_steps(),
                                         params.checkpoint_secs(),
                                         params.per_rank_dir(),
                                         params.ckpt_dist_epochs(),
                                         params.ckpt_dist_steps());
  }
  if (proto_cb.has_save_model()) {
    const auto& params = proto_cb.save_model();
    return new lbann_callback_save_model(params.dir(),
                                         params.extension());
  }

  //////////////////////////////////////////////////////////////////
  // Profiling
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_summary()) {
    const auto& params = proto_cb.summary();
    return new lbann_callback_summary(summarizer,
                                      params.batch_interval(),
                                      params.mat_interval());
  }
  if (proto_cb.has_profiler()) {
    return new lbann_callback_profiler();
  }

  //////////////////////////////////////////////////////////////////
  // Debugging
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_debug()) {
    const auto& params = proto_cb.debug();
    std::set<execution_mode> modes;
    for (const auto& mode : parse_list<>(params.phase())) {
      if (mode == "train" || mode == "training") {
        modes.insert(execution_mode::training);
      } else if (mode == "validate" || mode == "validation") {
        modes.insert(execution_mode::validation);
      } else if (mode == "test" || mode == "testing") {
        modes.insert(execution_mode::testing);
      } else {
        LBANN_ERROR("invalid execution mode (" + mode + ")");
      }
    }
    return new lbann_callback_debug(modes, summarizer);
  }
  if (proto_cb.has_debug_io()) {
    const auto& params = proto_cb.debug_io();
    const auto& phase = params.phase();
    const auto& lvl = params.lvl();
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
    const auto& params = proto_cb.dump_weights();
    return new lbann_callback_dump_weights(params.basename());
  }
  if (proto_cb.has_dump_activations()) {
    const auto& params = proto_cb.dump_activations();
    const auto& layer_names = parse_list<>(params.layer_names());
    return new lbann_callback_dump_activations(params.basename(),
                                               params.interval(),
                                               layer_names);
  }
  if (proto_cb.has_dump_gradients()) {
    const auto& params = proto_cb.dump_gradients();
    return new lbann_callback_dump_gradients(params.basename(),
                                             params.interval());
  }
  if (proto_cb.has_dump_mb_indices()) {
    const auto& params = proto_cb.dump_mb_indices();
    return new lbann_callback_dump_minibatch_sample_indices(params.basename(),
                                                            params.interval());
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
    const auto& params = proto_cb.gradient_check();
    return new lbann_callback_gradient_check(params.step_size(),
                                             params.verbose(),
                                             params.fail_on_error());
  }

  //////////////////////////////////////////////////////////////////
  // GPU memory profiling
  //////////////////////////////////////////////////////////////////
  if (proto_cb.has_gpu_memory_usage()) {
    return new lbann_callback_gpu_memory_usage();
  }
  
  return nullptr;
}

lbann_summary* construct_summarizer(lbann_comm* comm,
                                    const lbann_data::Model& m) {
  lbann_summary *summary = nullptr;
  bool master = comm->am_world_master();
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);
    if (callback.has_summary()) {
      const lbann_data::CallbackSummary& c = callback.summary();
      if (master) {
        std::cout << "constructing summarizer with dir: " << c.dir() << std::endl;
      }

      //check to see if directory exists
      struct stat sb;
      if (! ( stat(c.dir().c_str(), &sb) == 0 && S_ISDIR(sb.st_mode) )) {
        if (master) {
          throw lbann_exception(
            std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
            "summary directory " + c.dir() + " does not exist");
        }
      }
      summary = new lbann_summary(c.dir(), comm);
    }
  }
  return summary;
}

} // namespace proto
} // namespace lbann
