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

#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/checkpoint.hpp" // Reuse the checkpoint naming scheme
#include "lbann/utils/memory.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/trainer_file_utils.hpp"

#include <callbacks.pb.h>

#include <string>

namespace lbann {
namespace callback {

namespace dump_weights_internal {

/** @brief Format for weight files. */
class FileFormat
  : public Cloneable<HasAbstractFunction<FileFormat>>
{
public:
  FileFormat() = default;
  FileFormat(const FileFormat&) = default;
  FileFormat(FileFormat&&) = default;
  virtual ~FileFormat() noexcept = default;

  /** @brief Write weight values to file. */
  virtual void write(const weights& w, const std::string& file) const = 0;
};

namespace {

class TextFileFormat final
  : public Cloneable<TextFileFormat, FileFormat>
{
public:
  TextFileFormat() = default;

  void write(const weights& w, const std::string& file) const final {

    // Try casting weights values and writing
    if (try_write<float>(w, file)) { return; }
    if (try_write<double>(w, file)) { return; }
#ifdef LBANN_HAS_HALF
    if (try_write<cpu_fp16>(w, file)) { return; }
#endif // LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU_FP16
    if (try_write<fp16>(w, file)) { return; }
#endif // LBANN_HAS_GPU_FP16

    // Could not figure out weights' data type
    LBANN_ERROR(
      "could not write weights \"",w.get_name(),"\" ",
      "to text file ",file);

  }

private:

  /** @brief Try casting weight values and writing to file.
   *
   *  If the weights data can be cast to @c TensorDataType, then it is
   *  saved to file in Elemental's ASCII format.
   *
   *  @returns Whether the weight values were saved to file.
   */
  template <typename TensorDataType>
  bool try_write(const weights& w, const std::string& file) const {
    auto* typed_w = dynamic_cast<const data_type_weights<TensorDataType>*>(&w);
    if (typed_w == nullptr) {
      return false;
    }
    else {
      El::Write(typed_w->get_values(), file, El::ASCII);
      return true;
    }
  }

};

class BinaryFileFormat final
  : public Cloneable<BinaryFileFormat, FileFormat>
{
public:
  BinaryFileFormat() = default;

  void write(const weights& w, const std::string& file) const final {

    // Try casting weights values and writing
    if (try_write<float>(w, file)) { return; }
    if (try_write<double>(w, file)) { return; }
#ifdef LBANN_HAS_HALF
    if (try_write<cpu_fp16>(w, file)) { return; }
#endif // LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU_FP16
    if (try_write<fp16>(w, file)) { return; }
#endif // LBANN_HAS_GPU_FP16

    // Could not figure out weights' data type
    LBANN_ERROR(
      "could not write weights \"",w.get_name(),"\" ",
      "to text file ",file);

  }

private:

  /** @brief Try casting weight values and writing to file.
   *
   *  If the weights data can be cast to @c TensorDataType, then it is
   *  saved to file in Elemental's BINARY format.
   *
   *  @returns Whether the weight values were saved to file.
   */
  template <typename TensorDataType>
  bool try_write(const weights& w, const std::string& file) const {
    auto* typed_w = dynamic_cast<const data_type_weights<TensorDataType>*>(&w);
    if (typed_w == nullptr) {
      return false;
    }
    else {
      El::Write(typed_w->get_values(), file, El::BINARY);
      return true;
    }
  }

};

class DistributedBinaryFileFormat final
  : public Cloneable<DistributedBinaryFileFormat, FileFormat>
{
public:
  DistributedBinaryFileFormat() = default;

  void write(const weights& w, const std::string& file) const final {

    // Try casting weights values and writing
    if (try_write<float>(w, file)) { return; }
    if (try_write<double>(w, file)) { return; }
#ifdef LBANN_HAS_HALF
    if (try_write<cpu_fp16>(w, file)) { return; }
#endif // LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU_FP16
    if (try_write<fp16>(w, file)) { return; }
#endif // LBANN_HAS_GPU_FP16

    // Could not figure out weights' data type
    LBANN_ERROR(
      "could not write weights \"",w.get_name(),"\" ",
      "to text file ",file);

  }

private:

  /** @brief Try casting weight values and writing to file.
   *
   *  If the weights data can be cast to @c TensorDataType, then each
   *  non-redundant local matrix is saved to file in Elemental's
   *  BINARY format.
   *
   *  @returns Whether the weight values were saved to file.
   */
  template <typename TensorDataType>
  bool try_write(const weights& w, const std::string& file) const {
    auto* typed_w = dynamic_cast<const data_type_weights<TensorDataType>*>(&w);
    if (typed_w == nullptr) {
      return false;
    }
    else {
      const auto& mat = typed_w->get_values();
      if (mat.RedundantRank() == 0) {
        El::Write(
          mat.LockedMatrix(),
          El::BuildString(file, "_rank", mat.DistRank()),
          El::BINARY);
      }
      return true;
    }
  }

};

} // namespace <anon>

} // namespace dump_weights_internal

dump_weights::dump_weights(
  std::string dir,
  El::Int epoch_interval,
  std::unique_ptr<dump_weights_internal::FileFormat> file_format)
  : callback_base(),
    m_directory(std::move(dir)),
    m_epoch_interval(std::max(El::Int(1),epoch_interval)),
    m_file_format(std::move(file_format))
{}

dump_weights::dump_weights()
  : dump_weights("", 1, make_unique<dump_weights_internal::TextFileFormat>())
{}

dump_weights::dump_weights(const dump_weights& other)
  : callback_base(other),
    m_directory(other.m_directory),
    m_epoch_interval(other.m_epoch_interval),
    m_file_format(other.m_file_format->clone())
{}

dump_weights& dump_weights::operator=(const dump_weights& other) {
  callback_base::operator=(other);
  m_directory = other.m_directory;
  m_epoch_interval = other.m_epoch_interval;
  m_file_format = other.m_file_format->clone();
  return *this;
}

void dump_weights::on_train_begin(model *m) {
  do_dump_weights(*m, visitor_hook::execution_mode_begin);
}

void dump_weights::on_epoch_end(model *m) {
  const auto& context = static_cast<const sgd_execution_context&>(m->get_execution_context());
  if (context.get_epoch() % m_epoch_interval == 0) {
    do_dump_weights(*m, visitor_hook::epoch_end);
  }
}

void dump_weights::do_dump_weights(const model& m, visitor_hook hook) {
  const auto& context = static_cast<const sgd_execution_context&>(m.get_execution_context());
  const auto& t = context.get_trainer();

  // Create directory for weight files
  // Note: Use naming scheme from checkpoint callback
  std::string dir = El::BuildString(
    get_shared_checkpoint_dirname(
      t.get_name(),
      context.get_training_algorithm().get_name(),
      m_directory.c_str(),
      hook,
      context.get_execution_mode(),
      context.get_epoch(),
      context.get_step()),
    m.get_name(), '/');
  file::trainer_master_make_directory(dir, m.get_comm());

  // Save weights
  for (auto* w : m.get_weights()) {
    m_file_format->write(*w, El::BuildString(dir, w->get_name()));
  }

  // Update checkpoint file
  if (m.get_comm()->am_trainer_master()) {
    auto latest_file = get_last_shared_checkpoint_filename(
      t.get_name(),
      context.get_training_algorithm().get_name(),
      m_directory.c_str());
    write_latest(
      latest_file,
      hook,
      context.get_execution_mode(),
      context.get_epoch(),
      context.get_step());
  }

}

std::unique_ptr<callback_base>
build_dump_weights_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDumpWeights&>(proto_msg);

  // Initialize file format
  /// @todo Support binary and distributed binary
  std::unique_ptr<dump_weights_internal::FileFormat> file_format;
  if (params.format().empty() || params.format() == "text") {
    file_format = make_unique<dump_weights_internal::TextFileFormat>();
  }
  if (params.format() == "binary") {
    file_format = make_unique<dump_weights_internal::BinaryFileFormat>();
  }
  if (params.format() == "distributed_binary") {
    file_format = make_unique<dump_weights_internal::DistributedBinaryFileFormat>();
  }
  if (file_format == nullptr) {
    LBANN_ERROR("unrecognized file format \"",params.format(),"\"");
  }

  // Construct callback
  return make_unique<dump_weights>(
    params.directory(),
    params.epoch_interval(),
    std::move(file_format));

}

} // namespace callback
} // namespace lbann

CEREAL_REGISTER_TYPE_WITH_NAME(
  ::lbann::callback::dump_weights,
  "callback::dump_weights")
