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

#define LBANN_DATA_TYPE_WEIGHTS_INSTANTIATE
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/onnx_utils.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

#include "lbann/proto/layers.pb.h"
#include "lbann/proto/weights.pb.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

/** @brief Get a string version of tensor dimensions */
std::string stringify_dims(const std::vector<size_t>& dims)
{
  std::ostringstream oss;
  for (size_t i = 0; i < dims.size(); ++i)
    oss << (i > 0 ? "x" : "") << dims[i];
  return oss.str();
}

/** @brief Get string describing tensor dimensions.
 *  The tensor is stored in a matrix, although there may be multiple
 *  dimensions corresponding to the matrix height and width.
 */
std::string get_dims_string(const std::vector<size_t>& matrix_height_dims,
                            const std::vector<size_t>& matrix_width_dims)
{
  std::ostringstream oss;
  oss << "(" << stringify_dims(matrix_height_dims) << ")x"
      << "(" << stringify_dims(matrix_width_dims) << ")";
  return oss.str();
}

} // namespace

namespace lbann {

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights(lbann_comm& comm)
  : BaseType(comm)
{}

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights(const WeightsType& other)
  : BaseType(other)
{

  // Deep copies
  m_values.reset(other.m_values ? other.m_values->Copy() : nullptr);
  m_initializer =
    (other.m_initializer ? other.m_initializer->clone() : nullptr);
  m_optimizer = (other.m_optimizer ? other.m_optimizer->clone() : nullptr);
  if (m_optimizer != nullptr) {
    m_optimizer->set_weights(this);
  }
}

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::operator=(const WeightsType& other)
  -> WeightsType&
{
  weights::operator=(other);

  // Deep copies
  m_values.reset(other.m_values ? other.m_values->Copy() : nullptr);
  m_initializer =
    (other.m_initializer ? other.m_initializer->clone() : nullptr);
  m_optimizer = (other.m_optimizer ? other.m_optimizer->clone() : nullptr);
  if (m_optimizer != nullptr) {
    m_optimizer->set_weights(this);
  }

  return *this;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::do_augment_description_(
  description& desc) const
{

  // Optimizer
  if (m_optimizer != nullptr) {
    desc.add(m_optimizer->get_description());
  }

  // Initializer
  if (m_initializer != nullptr) {
    desc.add(m_initializer->get_description());
  }
}

// -----------------------------------------------
// Dimension accessors
// -----------------------------------------------
template <typename TensorDataType>
void data_type_weights<TensorDataType>::do_set_dims_(
  std::vector<size_t> const& matrix_height_dims,
  std::vector<size_t> const& matrix_width_dims)
{
  if (m_values != nullptr) {
    const El::Int height = this->get_matrix_height();
    const El::Int width = this->get_matrix_width();
    if (m_values->Height() != height || m_values->Width() != width) {
      LBANN_ERROR("attempted to set weights \"",
                  this->get_name(),
                  "\" "
                  "with dimensions ",
                  get_dims_string(matrix_height_dims, matrix_width_dims),
                  ", "
                  "but it is already setup with a ",
                  m_values->Height(),
                  " x ",
                  m_values->Width(),
                  " "
                  "weights matrix");
    }
  }
}

// -----------------------------------------------
// Initializer accessors
// -----------------------------------------------

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_initializer() -> InitializerType*
{
  return const_cast<InitializerType*>(
    static_cast<const data_type_weights&>(*this).get_initializer());
}
template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_initializer() const
  -> const InitializerType*
{
  return m_initializer.get();
}
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_initializer(
  std::unique_ptr<weights_initializer>&& init)
{
  using InitializerPtrType = InitializerType*;
  // Verify the dynamic type is compatible
  if (init && dynamic_cast<InitializerPtrType>(init.get()))
    // Safely transfer the memory; both release() and reset() are
    // noexcept so this is memory-safe. The dynamic_cast in the if
    // statement verifies the dynamic type; no need to redo it.
    m_initializer.reset(static_cast<InitializerPtrType>(init.release()));
  else if (init)
    // The provided pointer was not null, but the dynamic_cast
    // failed. This is an error.
    LBANN_ERROR("Initializer has incompatible dynamic type.");
  else
    // The provided pointer was null. Set the held pointer to null.
    m_initializer.reset();
}

// -----------------------------------------------
// Optimizer accessors
// -----------------------------------------------

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_optimizer() -> OptimizerType*
{
  return const_cast<OptimizerType*>(
    static_cast<const WeightsType&>(*this).get_optimizer());
}
template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_optimizer() const
  -> const OptimizerType*
{
  if (this->is_frozen()) {
    return nullptr;
  }
  else {
    return m_optimizer.get();
  }
}
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_optimizer(
  std::unique_ptr<optimizer>&& opt)
{
  using OptimizerPtrType = OptimizerType*;
  if (opt && dynamic_cast<OptimizerPtrType>(opt.get()))
    m_optimizer.reset(static_cast<OptimizerPtrType>(opt.release()));
  else if (opt)
    LBANN_ERROR("Optimizer has incompatible dynamic type");
  else
    m_optimizer.reset();
}

// -----------------------------------------------
// Setup
// -----------------------------------------------

template <typename TensorDataType>
void data_type_weights<TensorDataType>::do_setup_()
{

  // Return immediately if possible
  if (m_values != nullptr) {
    return;
  }

  // Construct matrix for weight values
  // If sharded, use STAR_VC distribution (column distributed) or VC_STAR (row
  // distributed) if width=1.
  auto matrix_dist = this->get_matrix_distribution();
  bool must_use_vc_star = (this->get_matrix_width() == 1);
  m_values.reset(AbsDistMatrixType::Instantiate(
    *matrix_dist.grid,
    matrix_dist.root,
    this->is_sharded() ? (must_use_vc_star ? El::VC : El::STAR)
                       : matrix_dist.colDist,
    this->is_sharded() ? (must_use_vc_star ? El::STAR : El::VC)
                       : matrix_dist.rowDist,
    (matrix_dist.blockHeight == 1 && matrix_dist.blockWidth == 1 ? El::ELEMENT
                                                                 : El::BLOCK),
    matrix_dist.device));

  if (this->is_sharded()) {
    m_values_view.reset(AbsDistMatrixType::Instantiate(
      *matrix_dist.grid,
      matrix_dist.root,
      matrix_dist.colDist,
      matrix_dist.rowDist,
      (matrix_dist.blockHeight == 1 && matrix_dist.blockWidth == 1 ? El::ELEMENT
                                                                   : El::BLOCK),
      matrix_dist.device));

#ifdef LBANN_HAS_GPU
    if (matrix_dist.device == El::Device::GPU) {
      m_values_view->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // LBANN_HAS_GPU
  }

  // Allocate memory
#ifdef LBANN_HAS_GPU
  if (matrix_dist.device == El::Device::GPU) {
    int memory_mode = 0; // Direct allocation
    m_values->Matrix().SetMemoryMode(memory_mode);
  }
#endif // LBANN_HAS_GPU
  m_values->AlignWith(matrix_dist);
  m_values->Resize(this->get_matrix_height(), this->get_matrix_width());

  // Initialize values
  if (m_initializer != nullptr) {
    m_initializer->fill(*m_values);
  }
  else {
    El::Zero(*m_values);
  }

  // Setup optimizer
  if (m_optimizer != nullptr) {
    m_optimizer->setup(this);
  }
}

// -----------------------------------------------
// Weight matrix accessors
// -----------------------------------------------

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_values() const
  -> const AbsDistMatrixType&
{
  auto matrix_dist = this->get_matrix_distribution();
  // If the weights are not sharded, return m_values directly
  if (!this->is_sharded()) {
    return *m_values;
  }

  // Get weights as necessary
  request_full_weights_async();
  wait_for_full_weights();

  return *m_values_view;
}

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_values_sharded()
  -> AbsDistMatrixType&
{
  return const_cast<AbsDistMatrixType&>(
    static_cast<const data_type_weights&>(*this).get_values_sharded());
}
template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_values_sharded() const
  -> const AbsDistMatrixType&
{
  if (m_values == nullptr) {
    LBANN_ERROR("attempted to access value shard of "
                "weights \"" +
                this->get_name() +
                "\" "
                "before they are set up");
  }
  return *m_values;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_values(
  const AbsDistMatrixType& values)
{
  if ((values.Height() != m_values->Height()) ||
      (values.Width() != m_values->Width())) {
    LBANN_ERROR("Expected matrix size ",
                this->get_matrix_height(),
                "x",
                this->get_matrix_width(),
                "; got a matrix with size ",
                values.Height(),
                "x",
                values.Width());
  }
  El::Copy(values, *m_values);
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value,
                                                  size_t index)
{

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  const auto& size = weights::get_size();
  if (index < 0 || index >= size) {
    LBANN_ERROR("attempted to set value in "
                "weights \"",
                this->get_name(),
                "\""
                "at index ",
                index,
                ", "
                "but there are ",
                size,
                " values");
  }
#endif // LBANN_DEBUG

  // Set matrix entry
  const auto& height = this->get_matrix_height();
  set_value(value, index % height, index / height);
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value,
                                                  std::vector<size_t> pos)
{

  // Get tensor dimensions
  const auto& dims = this->get_dims();

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  bool valid = dims.size() == pos.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    valid = valid && pos[i] >= 0 && pos[i] < dims[i];
  }
  if (!valid) {
    LBANN_ERROR("attempted to set value in "
                "weights \"",
                this->get_name(),
                "\""
                "at position (",
                stringify_dims(pos),
                ") "
                "in a tensor with dimensions ",
                stringify_dims(dims));
  }
#endif // LBANN_DEBUG

  // Get index of weight value and set
  size_t index = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    index = index * dims[i] + pos[i];
  }
  set_value(value, index);
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value,
                                                  size_t row,
                                                  size_t col)
{

#ifdef LBANN_DEBUG
  // Check that matrix entry is valid
  const auto& height = this->get_matrix_height();
  const auto& width = this->get_matrix_width();
  if (row < 0 || row >= height || col < 0 || col > width) {
    LBANN_ERROR("attempted to set weights value "
                "in weights \"",
                this->get_name(),
                "\""
                "at entry (",
                row,
                ",",
                col,
                ") "
                "in a ",
                height,
                "x",
                width,
                " matrix");
  }
#endif // LBANN_DEBUG

  // Set value if it is local
  auto& values = *m_values;
  if (values.IsLocal(row, col)) {
    values.SetLocal(values.LocalRow(row), values.LocalCol(col), value);
  }
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::request_full_weights_async() const
{
  m_values_view->AlignWith(this->get_matrix_distribution());
  m_values_view->Resize(this->get_matrix_height(), this->get_matrix_width());
  El::Copy(*m_values, *m_values_view);
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::wait_for_full_weights() const
{
  // TODO(later): Wait for asynchronous gather, if necessary
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::release_full_weights() const
{
  if (this->is_sharded()) {
    m_values_view->Empty();
  }
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::reconcile_values()
{
  LBANN_ERROR("This should not be called. Method is a candidate for removal");
  auto& values = *m_values;
  if (values.RedundantSize() > 1) {
    El::Scale(TensorDataType(1. / values.RedundantSize()), values);
    this->get_comm().allreduce(values, values.RedundantComm());
  }
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::reconcile_values(Al::request& req)
{
  LBANN_ERROR("This should not be called. Method is a candidate for removal");
  auto& values = *m_values;
  if (values.RedundantSize() > 1) {
    El::Scale(TensorDataType(1. / values.RedundantSize()), values);
    this->get_comm().nb_allreduce(values, values.RedundantComm(), req);
  }
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::write_proto(
  lbann_data::Weights& proto) const
{
  proto.Clear();
  proto.set_name(this->get_name());
  if (this->has_optimizer() == false)
    proto.mutable_optimizer()->mutable_no_optimizer();
  else
    this->get_optimizer()->write_proto(*proto.mutable_optimizer());

  this->get_initializer()->write_proto(*proto.mutable_initializer());

  proto.set_datatype(proto::TypeToProtoDataType<TensorDataType>::value);
}

#ifdef LBANN_HAS_ONNX
template <typename T>
using ADM = El::AbstractDistMatrix<T>;

template <typename T>
void data_type_weights<T>::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto* initializer = graph.add_initializer();
  auto const height_dims = this->get_matrix_height_dims();
  auto const width_dims = this->get_matrix_width_dims();

  serialize_to_onnx(this->get_values(), height_dims, width_dims, *initializer);

  initializer->set_name(this->get_name());
  initializer->set_doc_string(this->get_name() + " tensor values");
}
#endif // LBANN_HAS_ONNX

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_save(
  std::string const& ckpt_dir,
  std::vector<std::string> const& weight_list,
  El::FileFormat el_mode)
{

  std::string suffix = ".bin";
  if (el_mode == El::ASCII) {
    suffix = ".txt";
  }
  // create weight file name to match to weight list entry
  // Note that the prefix model_ has to be explicitly appended since
  // the persist class appends that string in the normal checkpoint functions
  auto l_name = El::BuildString("model_weights_",
                                this->get_name(),
                                "_",
                                m_values->Height(),
                                "x",
                                m_values->Width(),
                                suffix);

  auto it = std::find(weight_list.begin(), weight_list.end(), l_name);
  // If match is found read in weight values.
  if (it != weight_list.end()) {
    std::string full_path = ckpt_dir + *it;
    if (this->get_comm().am_world_master()) {
      std::cout << "Loading " << this->get_name() << " <- " << *it << "\n";
    }
    // check whether file exists
    int exists = lbann::exists(full_path.c_str());
    if (!exists) {
      throw lbann_exception(std::string("Failed to read weight matrix: ") +
                            full_path);
      return false;
    }
    El::Read(*m_values, full_path, el_mode, true);
  }
  return true;
}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_save(
  std::string const& ckpt_dir,
  std::vector<std::string> const& weight_list)
{
  load_from_save(ckpt_dir, weight_list, El::BINARY);
  load_from_save(ckpt_dir, weight_list, El::ASCII);
  return true;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::do_move_values_(
  data_type_weights& other)
{
  m_values = std::move(other.m_values);
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::do_steal_values_(weights& other)
{
  do_move_values_(dynamic_cast<data_type_weights<TensorDataType>&>(other));
}

} // namespace lbann

#define PROTO(T) template class lbann::data_type_weights<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

#define LBANN_CLASS_NAME data_type_weights
#include <lbann/macros/register_template_class_with_cereal.hpp>
