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
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/confusion_matrix.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/profiling.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace lbann {
namespace callback {

// ---------------------------------------------------------
// Constructors
// ---------------------------------------------------------

confusion_matrix::confusion_matrix(std::string&& prediction_layer,
                                   std::string&& label_layer,
                                   std::string&& prefix)
  : callback_base(1),
    m_prediction_layer(std::move(prediction_layer)),
    m_label_layer(std::move(label_layer)),
    m_prefix(std::move(prefix))
{}

confusion_matrix::confusion_matrix(std::string const& prediction_layer,
                                   std::string const& label_layer,
                                   std::string const& prefix)
  : callback_base(1),
    m_prediction_layer(prediction_layer),
    m_label_layer(label_layer),
    m_prefix(prefix)
{}

confusion_matrix::confusion_matrix(const confusion_matrix& other)
  : callback_base(other),
    m_prediction_layer(other.m_prediction_layer),
    m_label_layer(other.m_label_layer),
    m_prefix(other.m_prefix),
    m_counts(other.m_counts),
    m_predictions_v(other.m_predictions_v ? other.m_predictions_v->Copy()
                                          : nullptr),
    m_labels_v(other.m_labels_v ? other.m_labels_v->Copy() : nullptr)
{}

confusion_matrix& confusion_matrix::operator=(const confusion_matrix& other)
{
  callback_base::operator=(other);
  m_prediction_layer = other.m_prediction_layer;
  m_label_layer = other.m_label_layer;
  m_prefix = other.m_prefix;
  m_counts = other.m_counts;
  m_predictions_v.reset(other.m_predictions_v ? other.m_predictions_v->Copy()
                                              : nullptr);
  m_labels_v.reset(other.m_labels_v ? other.m_labels_v->Copy() : nullptr);
  return *this;
}

// ---------------------------------------------------------
// Setup
// ---------------------------------------------------------

void confusion_matrix::setup(model* m)
{
  callback_base::setup(m);

  // Initialize matrix views/copies
  const auto& predictions = get_predictions(*m);
  const auto& labels = get_labels(*m);
  auto dist_data = predictions.DistData();
  dist_data.device = El::Device::CPU;
  m_predictions_v.reset(AbsDistMatType::Instantiate(dist_data));
  m_labels_v.reset(AbsDistMatType::Instantiate(dist_data));

  // Check output dimensions of prediction and label layers
  if (predictions.Height() != labels.Height()) {
    LBANN_ERROR("callback \"",
                name(),
                "\" "
                "has prediction and label layers with different dimensions "
                "(prediction layer \"",
                m_prediction_layer,
                "\" "
                "outputs ",
                predictions.Height(),
                " entries, "
                "label layer \"",
                m_label_layer,
                "\" "
                "outputs ",
                labels.Height(),
                " entries)");
  }
}

// ---------------------------------------------------------
// Matrix access functions
// ---------------------------------------------------------

auto confusion_matrix::get_predictions(const model& m) const
  -> const AbsDistMatType&
{
  for (const auto* l : m.get_layers()) {
    if (l->get_name() == m_prediction_layer) {
      auto const& dtl = dynamic_cast<data_type_layer<DataType> const&>(*l);
      return dtl.get_activations();
    }
  }
  LBANN_ERROR("callback \"",
              name(),
              "\" could not find "
              "prediction layer \"",
              m_prediction_layer,
              "\"");
}

auto confusion_matrix::get_labels(const model& m) const -> const AbsDistMatType&
{
  for (const auto* l : m.get_layers()) {
    if (l->get_name() == m_label_layer) {
      auto const& dtl = dynamic_cast<data_type_layer<DataType> const&>(*l);
      return dtl.get_activations();
    }
  }
  LBANN_ERROR("callback \"",
              name(),
              "\" could not find "
              "label layer \"",
              m_prediction_layer,
              "\"");
}

// ---------------------------------------------------------
// Count management functions
// ---------------------------------------------------------

void confusion_matrix::reset_counts(const model& m)
{
  const auto& c = m.get_execution_context();
  auto& counts = m_counts[c.get_execution_mode()];
  const auto& num_classes = get_predictions(m).Height();
  counts.assign(num_classes * num_classes, 0);
}

void confusion_matrix::update_counts(const model& m)
{
  LBANN_CALIPER_MARK_FUNCTION;
  constexpr DataType zero = 0;

  // Get predictions
  const auto& predictions = get_predictions(m);
  const auto& num_classes = predictions.Height();
  m_predictions_v->Empty(false);
  m_predictions_v->AlignWith(predictions);
  if (m_predictions_v->DistData() == predictions.DistData()) {
    El::LockedView(*m_predictions_v, predictions);
  }
  else {
    El::Copy(predictions, *m_predictions_v);
  }
  const auto& local_predictions = m_predictions_v->LockedMatrix();

  // Get labels
  const auto& labels = get_labels(m);
  m_labels_v->Empty(false);
  m_labels_v->AlignWith(predictions);
  if (m_labels_v->DistData() == labels.DistData()) {
    El::LockedView(*m_labels_v, labels);
  }
  else {
    El::Copy(labels, *m_labels_v);
  }
  const auto& local_labels = m_labels_v->LockedMatrix();

  // Update counts
  const auto& c = m.get_execution_context();
  auto& counts = m_counts[c.get_execution_mode()];
  const auto& local_height = local_predictions.Height();
  const auto& local_width = local_predictions.Width();
  for (El::Int local_col = 0; local_col < local_width; ++local_col) {
    El::Int prediction_index = -1, label_index = -1;
    LBANN_OMP_PARALLEL_FOR
    for (El::Int local_row = 0; local_row < local_height; ++local_row) {
      if (local_predictions(local_row, local_col) != zero) {
        prediction_index = m_predictions_v->GlobalRow(local_row);
      }
      if (local_labels(local_row, local_col) != zero) {
        label_index = m_labels_v->GlobalRow(local_row);
      }
    }
    if (prediction_index >= 0 && label_index >= 0) {
      counts[label_index + prediction_index * num_classes]++;
    }
  }
}

void confusion_matrix::save_confusion_matrix(const model& m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m.get_execution_context());

  // Get counts
  const auto& mode = c.get_execution_mode();
  auto& counts = m_counts[mode];

  // Accumulate counts in master process
  // Note: Counts in non-root processes are set to zero, so this can
  // be called multiple times without affecting correctness.
  auto&& comm = *m.get_comm();
  if (comm.am_trainer_master()) {
    comm.trainer_reduce(static_cast<El::Int*>(MPI_IN_PLACE),
                        counts.size(),
                        counts.data());
  }
  else {
    comm.trainer_reduce(counts.data(),
                        counts.size(),
                        comm.get_trainer_master(),
                        El::mpi::SUM);
    counts.assign(counts.size(), 0);
  }

  // Save confusion matrix on master process
  if (comm.am_trainer_master()) {
    const auto& num_classes = get_predictions(m).Height();
    const auto& total_count = std::accumulate(counts.begin(), counts.end(), 0);
    const auto& scale = DataType(1) / total_count;

    // Construct output file name
    std::string mode_string;
    switch (mode) {
    case execution_mode::training:
      mode_string = "train-epoch" + std::to_string(c.get_epoch());
      break;
    case execution_mode::validation:
      mode_string = "validation-epoch" + std::to_string(c.get_epoch());
      break;
    case execution_mode::testing:
      mode_string = "test";
      break;
    default:
      return; // Exit immediately if execution mode is unknown
    }

    // Write to file
    std::ofstream fs(m_prefix + mode_string + ".csv");
    for (El::Int i = 0; i < num_classes; ++i) {
      for (El::Int j = 0; j < num_classes; ++j) {
        fs << (j > 0 ? "," : "") << counts[j + i * num_classes] * scale;
      }
      fs << "\n";
    }
    fs.close();
  }
}

// ---------------------------------------------------------
// Protobuf Serialization
// ---------------------------------------------------------

void confusion_matrix::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_confusion_matrix();
  msg->set_prediction(m_prediction_layer);
  msg->set_label(m_label_layer);
  msg->set_prefix(m_prefix);
}

std::unique_ptr<callback_base> build_confusion_matrix_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackConfusionMatrix&>(
      proto_msg);
  return std::make_unique<confusion_matrix>(params.prediction(),
                                            params.label(),
                                            params.prefix());
}

} // namespace callback
} // namespace lbann
