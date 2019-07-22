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

// Get the declarations of all the builders for registration
#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/callback_check_dataset.hpp"
#include "lbann/callbacks/callback_check_gradients.hpp"
#include "lbann/callbacks/callback_check_init.hpp"
#include "lbann/callbacks/callback_check_metric.hpp"
#include "lbann/callbacks/callback_checknan.hpp"
#include "lbann/callbacks/callback_checkpoint.hpp"
#include "lbann/callbacks/callback_checksmall.hpp"
#include "lbann/callbacks/callback_confusion_matrix.hpp"
#include "lbann/callbacks/callback_debug.hpp"
#include "lbann/callbacks/callback_debug_io.hpp"
#include "lbann/callbacks/callback_dump_error_signals.hpp"
#include "lbann/callbacks/callback_dump_gradients.hpp"
#include "lbann/callbacks/callback_dump_minibatch_sample_indices.hpp"
#include "lbann/callbacks/callback_dump_outputs.hpp"
#include "lbann/callbacks/callback_dump_weights.hpp"
#include "lbann/callbacks/callback_early_stopping.hpp"
#include "lbann/callbacks/callback_gpu_memory_usage.hpp"
#include "lbann/callbacks/callback_hang.hpp"
#include "lbann/callbacks/callback_imcomm.hpp"
#include "lbann/callbacks/callback_io.hpp"
#include "lbann/callbacks/callback_learning_rate.hpp"
#include "lbann/callbacks/callback_ltfb.hpp"
#include "lbann/callbacks/callback_mixup.hpp"
#include "lbann/callbacks/callback_perturb_adam.hpp"
#include "lbann/callbacks/callback_perturb_dropout.hpp"
#include "lbann/callbacks/callback_print.hpp"
#include "lbann/callbacks/callback_replace_weights.hpp"
#include "lbann/callbacks/callback_save_images.hpp"
#include "lbann/callbacks/callback_save_model.hpp"
#include "lbann/callbacks/callback_save_topk_models.hpp"
#include "lbann/callbacks/callback_summary.hpp"
#include "lbann/callbacks/callback_sync_layers.hpp"
#include "lbann/callbacks/callback_sync_selected.hpp"
#include "lbann/callbacks/callback_timeline.hpp"
#include "lbann/callbacks/callback_timer.hpp"
#include "lbann/callbacks/callback_variable_minibatch.hpp"

#include "lbann/proto/factories.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/memory.hpp"

#include <google/protobuf/message.h>

#include <functional>
#include <memory>
#include <string>

namespace lbann {
namespace proto {
namespace {


// Define the factory type.
using factory_type = lbann::generic_factory<
  lbann_callback,
  std::string,
  generate_builder_type<lbann_callback,
                        google::protobuf::Message const&,
                        lbann_summary*>,
  default_key_error_policy>;

void register_default_builders(factory_type& factory)
{
  factory.register_builder("CallbackAdaptiveLearningRate",
                           build_callback_adaptive_learning_rate_from_pbuf);
  factory.register_builder("CallbackCheckDataset",
                           build_callback_check_dataset_from_pbuf);
  factory.register_builder("CallbackCheckGradients",
                           build_callback_check_gradients_from_pbuf);
  factory.register_builder("CallbackCheckInit",
                           build_callback_check_init_from_pbuf);
  factory.register_builder("CallbackCheckMetric",
                           build_callback_check_metric_from_pbuf);
  factory.register_builder("CallbackCheckNaN",
                           build_callback_check_nan_from_pbuf);
  factory.register_builder("CallbackCheckpoint",
                           build_callback_checkpoint_from_pbuf);
  factory.register_builder("CallbackCheckSmall",
                           build_callback_check_small_from_pbuf);
  factory.register_builder("CallbackConfusionMatrix",
                           build_callback_confusion_matrix_from_pbuf);
  factory.register_builder("CallbackDebug",
                           build_callback_debug_from_pbuf);
  factory.register_builder("CallbackDebugIO",
                           build_callback_debug_io_from_pbuf);
  factory.register_builder("CallbackDispIOStats",
                           build_callback_disp_io_stats_from_pbuf);
  factory.register_builder("CallbackDropFixedLearningRate",
                           build_callback_drop_fixed_learning_rate_from_pbuf);
  factory.register_builder("CallbackDumpErrorSignals",
                           build_callback_dump_error_signals_from_pbuf);
  factory.register_builder("CallbackDumpGradients",
                           build_callback_dump_gradients_from_pbuf);
  factory.register_builder("CallbackDumpMBIndices",
                           build_callback_dump_mb_indices_from_pbuf);
  factory.register_builder("CallbackDumpOutputs",
                           build_callback_dump_outputs_from_pbuf);
  factory.register_builder("CallbackDumpWeights",
                           build_callback_dump_weights_from_pbuf);
  factory.register_builder("CallbackEarlyStopping",
                           build_callback_early_stopping_from_pbuf);
  factory.register_builder("CallbackGPUMemoryUsage",
                           build_callback_gpu_memory_usage_from_pbuf);
  factory.register_builder("CallbackHang",
                           build_callback_hang_from_pbuf);
  factory.register_builder("CallbackImComm",
                           build_callback_imcomm_from_pbuf);
  factory.register_builder(
    "CallbackLinearGrowthLearningRate",
    build_callback_linear_growth_learning_rate_from_pbuf);
  factory.register_builder("CallbackLTFB",
                           build_callback_ltfb_from_pbuf);
  factory.register_builder("CallbackMinibatchSchedule",
                           build_callback_minibatch_schedule_from_pbuf);
  factory.register_builder("CallbackMixup",
                           build_callback_mixup_from_pbuf);
  factory.register_builder(
    "CallbackOptimizerwiseAdaptiveLearningRate",
    build_callback_optimizerwise_adaptive_learning_rate_from_pbuf);
  factory.register_builder("CallbackPerturbAdam",
                           build_callback_perturb_adam_from_pbuf);
  factory.register_builder("CallbackPerturbDropout",
                           build_callback_perturb_dropout_from_pbuf);
  factory.register_builder("CallbackPolyLearningRate",
                           build_callback_poly_learning_rate_from_pbuf);
  factory.register_builder("CallbackPrint",
                           build_callback_print_from_pbuf);
  factory.register_builder("CallbackProfiler",
                           build_callback_profiler_from_pbuf);
  factory.register_builder("CallbackReplaceWeights",
                           build_callback_replace_weights_from_pbuf);
  factory.register_builder("CallbackSaveImages",
                           build_callback_save_images_from_pbuf);
  factory.register_builder("CallbackSaveModel",
                           build_callback_save_model_from_pbuf);
  factory.register_builder("CallbackSaveTopKModels",
                           build_callback_save_topk_models_from_pbuf);
  factory.register_builder("CallbackStepLearningRate",
                           build_callback_step_learning_rate_from_pbuf);
  factory.register_builder("CallbackStepMinibatch",
                           build_callback_step_minibatch_from_pbuf);
  factory.register_builder("CallbackSummary",
                           build_callback_summary_from_pbuf);
  factory.register_builder("CallbackSyncLayers",
                           build_callback_sync_layers_from_pbuf);
  factory.register_builder("CallbackSyncSelected",
                           build_callback_sync_selected_from_pbuf);
  factory.register_builder("CallbackTimeline",
                           build_callback_timeline_from_pbuf);
  factory.register_builder("CallbackTimer",
                           build_callback_timer_from_pbuf);
}

// Manage a global factory
struct factory_manager
{
    factory_type factory_;

    factory_manager() {
        register_default_builders(factory_);
    }
};

factory_manager factory_mgr_;
factory_type const& get_callback_factory() noexcept
{
  return factory_mgr_.factory_;
}

} // namespace

std::unique_ptr<lbann_callback>
construct_callback(
  const google::protobuf::Message& proto_msg, lbann_summary* summarizer) {

  auto const& factory = get_callback_factory();
  auto const& msg =
    helpers::get_oneof_message(proto_msg, "callback_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg, summarizer);
}

lbann_summary* construct_summarizer(lbann_comm* comm,
                                    const lbann_data::Model& m) {
  lbann_summary *summary = nullptr;
  bool master = comm->am_world_master();
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);
    if (callback.has_summary()) {
      const lbann_data::Callback::CallbackSummary& c = callback.summary();
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
