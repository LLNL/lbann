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
////////////////////////////////////////////////////////////////////////////////

// Get the declarations of all the builders for registration
#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/check_dataset.hpp"
#include "lbann/callbacks/check_gradients.hpp"
#include "lbann/callbacks/check_init.hpp"
#include "lbann/callbacks/check_metric.hpp"
#include "lbann/callbacks/check_nan.hpp"
#include "lbann/callbacks/check_small.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/compute_model_size.hpp"
#include "lbann/callbacks/confusion_matrix.hpp"
#include "lbann/callbacks/debug.hpp"
#include "lbann/callbacks/debug_io.hpp"
#include "lbann/callbacks/dump_error_signals.hpp"
#include "lbann/callbacks/dump_gradients.hpp"
#include "lbann/callbacks/dump_minibatch_sample_indices.hpp"
#include "lbann/callbacks/dump_model_graph.hpp"
#include "lbann/callbacks/dump_outputs.hpp"
#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/early_stopping.hpp"
#ifdef LBANN_HAS_ONNX
#include "lbann/callbacks/export_onnx.hpp"
#endif // LBANN_HAS_ONNX
#include "lbann/callbacks/alternate_updates.hpp"
#include "lbann/callbacks/gpu_memory_usage.hpp"
#include "lbann/callbacks/hang.hpp"
#include "lbann/callbacks/imcomm.hpp"
#include "lbann/callbacks/learning_rate.hpp"
#include "lbann/callbacks/load_model.hpp"
#include "lbann/callbacks/ltfb.hpp"
#include "lbann/callbacks/mixup.hpp"
#include "lbann/callbacks/monitor_io.hpp"
#include "lbann/callbacks/perturb_adam.hpp"
#include "lbann/callbacks/perturb_dropout.hpp"
#include "lbann/callbacks/perturb_learning_rate.hpp"
#include "lbann/callbacks/perturb_weights.hpp"
#include "lbann/callbacks/print_model_description.hpp"
#include "lbann/callbacks/print_statistics.hpp"
#include "lbann/callbacks/profiler.hpp"
#include "lbann/callbacks/profiler_caliper.hpp"
#include "lbann/callbacks/replace_weights.hpp"
#include "lbann/callbacks/save_images.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/callbacks/save_topk_models.hpp"
#include "lbann/callbacks/set_weights_value.hpp"
#include "lbann/callbacks/summarize_images.hpp"
#include "lbann/callbacks/summary.hpp"
#include "lbann/callbacks/sync_layers.hpp"
#include "lbann/callbacks/timeline.hpp"
#include "lbann/callbacks/timer.hpp"
#include "lbann/callbacks/variable_minibatch.hpp"

#include "lbann/proto/factories.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/callbacks.pb.h"
#include "lbann/proto/model.pb.h"

#include <google/protobuf/message.h>

#include <memory>
#include <string>

namespace lbann {
namespace proto {

namespace {

// Define the factory type.
using factory_type =
  generic_factory<lbann::callback_base,
                  std::string,
                  generate_builder_type<callback_base,
                                        google::protobuf::Message const&,
                                        std::shared_ptr<lbann_summary> const&>,
                  default_key_error_policy>;

void register_default_builders(factory_type& factory)
{
  using namespace ::lbann::callback;
  factory.register_builder("CallbackAdaptiveLearningRate",
                           build_adaptive_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackAlternateUpdates",
                           build_alternate_updates_callback_from_pbuf);
  factory.register_builder("CallbackCheckDataset",
                           build_check_dataset_callback_from_pbuf);
  factory.register_builder("CallbackCheckGradients",
                           build_check_gradients_callback_from_pbuf);
  factory.register_builder("CallbackCheckInit",
                           build_check_init_callback_from_pbuf);
  factory.register_builder("CallbackCheckMetric",
                           build_check_metric_callback_from_pbuf);
  factory.register_builder("CallbackCheckNaN",
                           build_check_nan_callback_from_pbuf);
  factory.register_builder("CallbackCheckSmall",
                           build_check_small_callback_from_pbuf);
  factory.register_builder("CallbackConfusionMatrix",
                           build_confusion_matrix_callback_from_pbuf);
  factory.register_builder("CallbackComputeModelSize",
                           build_compute_model_size_callback_from_pbuf);
  factory.register_builder("CallbackDebug", build_debug_callback_from_pbuf);
  factory.register_builder("CallbackDebugIO",
                           build_debug_io_callback_from_pbuf);
  factory.register_builder("CallbackDispIOStats",
                           build_monitor_io_callback_from_pbuf);
  factory.register_builder("CallbackDropFixedLearningRate",
                           build_drop_fixed_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackDumpErrorSignals",
                           build_dump_error_signals_callback_from_pbuf);
  factory.register_builder("CallbackDumpGradients",
                           build_dump_gradients_callback_from_pbuf);
  factory.register_builder("CallbackDumpModelGraph",
                           build_dump_model_graph_callback_from_pbuf);
  factory.register_builder("CallbackDumpMBIndices",
                           build_dump_mb_indices_callback_from_pbuf);
  factory.register_builder("CallbackDumpOutputs",
                           build_dump_outputs_callback_from_pbuf);
  factory.register_builder("CallbackDumpWeights",
                           build_dump_weights_callback_from_pbuf);
  factory.register_builder("CallbackEarlyStopping",
                           build_early_stopping_callback_from_pbuf);
#ifdef LBANN_HAS_ONNX
  factory.register_builder("CallbackExportOnnx",
                           build_export_onnx_callback_from_pbuf);
#endif // LBANN_HAS_ONNX
  factory.register_builder("CallbackGPUMemoryUsage",
                           build_gpu_memory_usage_callback_from_pbuf);
  factory.register_builder("CallbackHang", build_hang_callback_from_pbuf);
  factory.register_builder("CallbackImComm", build_imcomm_callback_from_pbuf);
  factory.register_builder(
    "CallbackLinearGrowthLearningRate",
    build_linear_growth_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackLTFB", build_ltfb_callback_from_pbuf);
  factory.register_builder("CallbackMinibatchSchedule",
                           build_minibatch_schedule_callback_from_pbuf);
  factory.register_builder("CallbackMixup", build_mixup_callback_from_pbuf);
  factory.register_builder(
    "CallbackOptimizerwiseAdaptiveLearningRate",
    build_optimizerwise_adaptive_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackPerturbAdam",
                           build_perturb_adam_callback_from_pbuf);
  factory.register_builder("CallbackPerturbDropout",
                           build_perturb_dropout_callback_from_pbuf);
  factory.register_builder("CallbackPerturbLearningRate",
                           build_perturb_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackPerturbWeights",
                           build_perturb_weights_callback_from_pbuf);
  factory.register_builder("CallbackPolyLearningRate",
                           build_poly_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackPrintModelDescription",
                           build_print_model_description_callback_from_pbuf);
  factory.register_builder("CallbackPrint",
                           build_print_statistics_callback_from_pbuf);
  factory.register_builder("CallbackProfiler",
                           build_profiler_callback_from_pbuf);
  factory.register_builder("CallbackProfilerCaliper",
                           build_profiler_caliper_callback_from_pbuf);
  factory.register_builder("CallbackReplaceWeights",
                           build_replace_weights_callback_from_pbuf);
  factory.register_builder("CallbackSaveImages",
                           build_save_images_callback_from_pbuf);
  factory.register_builder("CallbackSaveModel",
                           build_save_model_callback_from_pbuf);
  factory.register_builder("CallbackLoadModel",
                           build_load_model_callback_from_pbuf);
  factory.register_builder("CallbackSaveTopKModels",
                           build_save_topk_models_callback_from_pbuf);
  factory.register_builder("CallbackStepLearningRate",
                           build_step_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackSetLearningRate",
                           build_set_learning_rate_callback_from_pbuf);
  factory.register_builder("CallbackStepMinibatch",
                           build_step_minibatch_callback_from_pbuf);
  factory.register_builder("CallbackSummarizeImages",
                           build_summarize_images_callback_from_pbuf);
  factory.register_builder("CallbackSummary", build_summary_callback_from_pbuf);
  factory.register_builder("CallbackSyncLayers",
                           build_sync_layers_callback_from_pbuf);
  factory.register_builder("CallbackTimeline",
                           build_timeline_callback_from_pbuf);
  factory.register_builder("CallbackTimer", build_timer_callback_from_pbuf);
  factory.register_builder("CallbackSetWeightsValue",
                           build_set_weights_value_callback_from_pbuf);
}

// Manage a global factory
struct factory_manager
{
  factory_type factory_;

  factory_manager() { register_default_builders(factory_); }
};

factory_manager factory_mgr_;
factory_type const& get_callback_factory() noexcept
{
  return factory_mgr_.factory_;
}

} // namespace

std::unique_ptr<callback_base>
construct_callback(const google::protobuf::Message& proto_msg,
                   std::shared_ptr<lbann_summary> const& summarizer)
{

  auto const& factory = get_callback_factory();
  auto const& msg = protobuf::get_oneof_message(proto_msg, "callback_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg, summarizer);
}

std::unique_ptr<lbann_summary> construct_summarizer(lbann_comm* comm,
                                                    const lbann_data::Model& m)
{
  const bool master = comm->am_world_master();
  if (m.has_summarizer()) {
    auto dir = m.summarizer().dir();

    if (master) {
      std::cout << "constructing summarizer with dir: " << dir << std::endl;
    }

    // check to see if directory exists
    if (!file::directory_exists(dir)) {
      LBANN_ERROR("summary directory ", dir, " does not exist.");
    }

    return std::make_unique<lbann_summary>(dir, comm);
  }
  return nullptr;
}

namespace {
// Define a second factory type for callbacks that don't require a summarizer
using factory_type_no_summarizer = lbann::generic_factory<
  lbann::callback_base,
  std::string,
  generate_builder_type<lbann::callback_base, google::protobuf::Message const&>,
  default_key_error_policy>;

void register_default_builders_no_summarizer(
  factory_type_no_summarizer& factory)
{
  using namespace callback;
  factory.register_builder("CallbackCheckpoint",
                           build_checkpoint_callback_from_pbuf);
}

// Manage a global factory with no summarizer
struct factory_manager_no_summarizer
{
  factory_type_no_summarizer factory_;

  factory_manager_no_summarizer()
  {
    register_default_builders_no_summarizer(factory_);
  }
};

factory_manager_no_summarizer factory_mgr_no_summarizer_;
factory_type_no_summarizer const& get_callback_factory_no_summarizer() noexcept
{
  return factory_mgr_no_summarizer_.factory_;
}

} // namespace

std::unique_ptr<callback_base>
construct_callback(const google::protobuf::Message& proto_msg)
{

  auto const& factory = get_callback_factory_no_summarizer();
  auto const& msg = protobuf::get_oneof_message(proto_msg, "callback_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg);
}
} // namespace proto
} // namespace lbann
