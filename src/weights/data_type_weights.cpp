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

#define LBANN_DATA_TYPE_WEIGHTS_INSTANTIATE
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/io/file_io.hpp"

#include <layers.pb.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace {

/** @brief Get a string version of tensor dimensions */
std::string stringify_dims(const std::vector<int>& dims)
{
  std::ostringstream oss;
  oss << dims.front();
  for (size_t i = 1; i < dims.size(); ++i)
    oss << "x" << dims[i];
  return oss.str();
}

/** @brief Get string describing tensor dimensions.
 *  The tensor is stored in a matrix, although there may be multiple
 *  dimensions corresponding to the matrix height and width.
 */
std::string get_dims_string(const std::vector<int>& matrix_height_dims,
                            const std::vector<int>& matrix_width_dims) {
  std::ostringstream oss;
  oss << "(" << stringify_dims(matrix_height_dims) << ")x"
      << "(" << stringify_dims(matrix_width_dims) << ")";
  return oss.str();
}

} // namespace

namespace lbann {

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights(lbann_comm* comm)
  : BaseType(comm) {}

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights(const WeightsType& other)
  : BaseType(other) {

  // Deep copies
  m_values.reset(other.m_values ? other.m_values->Copy() : nullptr);
  m_initializer = (other.m_initializer
                   ? other.m_initializer->clone() : nullptr);
  m_optimizer = (other.m_optimizer
                 ? other.m_optimizer->clone() : nullptr);
  if (m_optimizer != nullptr) {
    m_optimizer->set_weights(this);
  }

}

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::operator=(const WeightsType& other) -> WeightsType& {
  weights::operator=(other);

  // Deep copies
  m_values.reset(other.m_values ? other.m_values->Copy() : nullptr);
  m_initializer = (other.m_initializer
                   ? other.m_initializer->clone() : nullptr);
  m_optimizer = (other.m_optimizer
                 ? other.m_optimizer->clone() : nullptr);
  if (m_optimizer != nullptr) {
    m_optimizer->set_weights(this);
  }

  return *this;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::do_augment_description_(description& desc) const {

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
  std::vector<int> const& matrix_height_dims,
  std::vector<int> const& matrix_width_dims) {
  if (m_values != nullptr) {
    const auto& height = this->get_matrix_height();
    const auto& width = this->get_matrix_width();
    if (m_values->Height() != height || m_values->Width() != width) {
      LBANN_ERROR("attempted to set weights \"", this->get_name(), "\" "
                  "with dimensions ",
                  get_dims_string(matrix_height_dims, matrix_width_dims), ", "
                  "but it is already setup with a ",
                  m_values->Height(), " x ", m_values->Width(), " "
                  "weights matrix");
    }
  }
}

// -----------------------------------------------
// Initializer accessors
// -----------------------------------------------

template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_initializer() -> InitializerType* {
  return const_cast<InitializerType*>(static_cast<const data_type_weights&>(*this).get_initializer());
}
template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_initializer() const
  -> const InitializerType* {
  return m_initializer.get();
}
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_initializer(
  std::unique_ptr<weights_initializer>&& init) {
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
auto data_type_weights<TensorDataType>::get_optimizer()
  -> OptimizerType* {
  return const_cast<OptimizerType*>(
           static_cast<const WeightsType&>(*this).get_optimizer());
}
template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_optimizer() const
  -> const OptimizerType* {
  if (this->is_frozen()) {
    return nullptr;
  } else {
    return m_optimizer.get();
  }
}
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_optimizer(
  std::unique_ptr<optimizer>&& opt) {
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
void data_type_weights<TensorDataType>::do_setup_() {
  // Return immediately if weights have already been setup
  if (m_values != nullptr) { return; }

  auto matrix_dist = this->get_matrix_distribution();
  // Construct weights matrix
  m_values.reset(AbsDistMatrixType::Instantiate(*matrix_dist.grid,
                                                matrix_dist.root,
                                                matrix_dist.colDist,
                                                matrix_dist.rowDist,
                                                (matrix_dist.blockHeight == 1
                                                 && matrix_dist.blockWidth == 1 ?
                                                 El::ELEMENT : El::BLOCK),
                                                matrix_dist.device));
  m_values->AlignWith(matrix_dist);
  m_values->Resize(this->get_matrix_height(), this->get_matrix_width());
  if (m_initializer != nullptr) {
    m_initializer->fill(*m_values);
  } else {
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
auto data_type_weights<TensorDataType>::get_values() -> AbsDistMatrixType& {
  return const_cast<AbsDistMatrixType&>(static_cast<const data_type_weights&>(*this).get_values());
}
template <typename TensorDataType>
auto data_type_weights<TensorDataType>::get_values() const
  -> const AbsDistMatrixType& {
  if (m_values == nullptr) {
    LBANN_ERROR("attempted to access values of "
                "weights \"" + this->get_name() + "\" "
                "before they are setup");
  }
  return *m_values;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_values(const AbsDistMatrixType& values) {
  if ((values.Height() != get_values().Height())
      || (values.Width() != get_values().Width())) {
    LBANN_ERROR("Expected matrix size ",
                this->get_matrix_height(), "x", this->get_matrix_width(),
                "; got a matrix with size ",
                values.Height(), "x", values.Width());
  }
  El::Copy(values, get_values());
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value, int index) {

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  const auto& size = weights::get_size();
  if (index < 0 || index >= size) {
    LBANN_ERROR("attempted to set value in "
                "weights \"", this->get_name(), "\""
                "at index ", index, ", "
                "but there are ", size, " values");
  }
#endif // LBANN_DEBUG

  // Set matrix entry
  const auto& height = this->get_matrix_height();
  set_value(value, index % height, index / height);

}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value, std::vector<int> pos) {

  // Get tensor dimensions
  const auto& dims = this->get_dims();

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  bool valid = dims.size() == pos.size();
  for (size_t i = 0 ; i < dims.size(); ++i) {
    valid = valid && pos[i] >= 0 && pos[i] < dims[i];
  }
  if (!valid) {
    LBANN_ERROR("attempted to set value in "
                "weights \"", this->get_name(), "\""
                "at position (", stringify_dims(pos), ") "
                "in a tensor with dimensions ", stringify_dims(dims));
      }
#endif // LBANN_DEBUG

  // Get index of weight value and set
  int index = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    index = index * dims[i] + pos[i];
  }
  set_value(value, index);
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value, int row, int col) {

#ifdef LBANN_DEBUG
  // Check that matrix entry is valid
  const auto& height = this->get_matrix_height();
  const auto& width = this->get_matrix_width();
  if (row < 0 || row >= height || col < 0 || col > width ) {
    LBANN_ERROR("attempted to set weights value "
                "in weights \"", this->get_name(), "\""
                "at entry (", row, ",", col, ") "
                "in a ", height, "x", width, " matrix");
  }
#endif // LBANN_DEBUG

  // Set value if it is local
  auto& values = get_values();
  if (values.IsLocal(row, col)) {
    values.SetLocal(values.LocalRow(row), values.LocalCol(col), value);
  }

}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::reconcile_values() {
  auto& values = get_values();
  if (values.RedundantSize() > 1) {
    El::Scale(TensorDataType(1. / values.RedundantSize()), values);
    this->get_comm().allreduce(values, values.RedundantComm());
  }
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::reconcile_values(Al::request& req) {
  auto& values = get_values();
  if (values.RedundantSize() > 1) {
    El::Scale(TensorDataType(1. / values.RedundantSize()), values);
    this->get_comm().nb_allreduce(values, values.RedundantComm(), req);
  }
}

// -----------------------------------------------
// Checkpointing
// -----------------------------------------------

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::save_to_checkpoint_shared(lbann::persist& p)
{
  // define name to store weight values
  char l_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld", this->get_name().c_str(), m_values->Height(), m_values->Width());
  // write weights using persist call -- uses Elemental's write function.
  p.write_distmat(persist_type::model, l_name, m_values.get());
  // if saving training state, also write out state of optimizer
  if (m_optimizer != nullptr && (p.get_cb_type() != callback_type::weights_only)) {
    m_optimizer->save_to_checkpoint_shared(p, this->get_name());
  }

  return true;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::write_proto(lbann_data::WeightsData* proto) const {

  // Set proto properties
  proto->Clear();
  proto->set_name(this->get_name());
  for (const auto& d : this->get_dims()) {
    proto->mutable_shape()->add_dim(d);
  }
  proto->set_height(this->get_matrix_height());
  proto->set_width(this->get_matrix_width());

  // Write weight values to prototext on world master process
  CircMatDT<TensorDataType, El::Device::CPU> values = *m_values; /// @todo What if weights are on GPU?
  values.SetRoot(0); /// @todo What if world master is not process 0?
  if (this->get_comm().am_world_master()) {
    const auto& local_values = values.LockedMatrix();
    const El::Int height = local_values.Height();
    const El::Int width = local_values.Width();
    /// @todo OpenMP parallelization
    /** @todo Our matrices are column-major while Numpy expects
     *  row-major matrices. This row-wise iteration is fine for
     *  matrices and column vectors, but it can mess up the order of
     *  the weights if a high-dimensional tensor is represented as a
     *  matrix. This is what we need for quantization on convolution
     *  kernel weights.
     */
    for (El::Int i = 0; i < height; ++i) {
      for (El::Int j = 0; j < width; ++j) {
        proto->add_data(local_values(i,j));
      }
    }
  }

}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_checkpoint_shared(lbann::persist& p)
{
  // define filename containing saved weight values
  auto f_name = El::BuildString("weights_", this->get_name(), "_",
                                m_values->Height(), "x", m_values->Width(),
                                ".bin");
  p.read_distmat(persist_type::model, f_name.c_str(), m_values.get());
  if (m_optimizer != nullptr) {
    m_optimizer->load_from_checkpoint_shared(p, this->get_name());
  }

  return true;
}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_save(std::string const& ckpt_dir,
                                                       std::vector<std::string> const& weight_list){
  // create weight file name to match to weight list entry
  auto l_name = El::BuildString("model_weights_", this->get_name(), "_",
                                m_values->Height(), "x", m_values->Width(), ".bin");
  auto it = std::find(weight_list.begin(),weight_list.end(),l_name);
  // If match is found read in weight values.
  if(it != weight_list.end()) {
    std::string full_path = ckpt_dir + *it;
    if(this->get_comm().am_world_master()) {
      std::cout << "Loading " << this->get_name() << " <- " << *it << "\n";
    }
    // check whether file exists
    int exists = lbann::exists(full_path.c_str());
    if (! exists) {
      throw lbann_exception(std::string("Failed to read weight matrix: ") + full_path);
      return false;
    }
    El::Read(*m_values,full_path, El::BINARY, true);
  }
  return true;
}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::save_to_checkpoint_distributed(lbann::persist& p){
  // Functions identically to shared checkpoint except weights and parameters are saved on a per rank basis
  auto l_name = El::BuildString("weights_", this->get_name(),
                                "_", m_values->LocalHeight(),
                                "x", m_values->LocalWidth(), ".bin");
  p.write_rank_distmat(persist_type::model, l_name.c_str(), *m_values);
  if (m_optimizer != nullptr) {
    m_optimizer->save_to_checkpoint_distributed(p, this->get_name());
  }
  return true;
}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_checkpoint_distributed(lbann::persist& p){
  // Functions identically to shared checkpoint except weights and parameters are loaded on a per rank basis
  auto l_name = El::BuildString("weights_", this->get_name(),
                                "_", m_values->LocalHeight(),
                                "x", m_values->LocalWidth(), ".bin");
  p.read_rank_distmat(persist_type::model, l_name.c_str(), *m_values);
  if (m_optimizer != nullptr) {
    m_optimizer->load_from_checkpoint_distributed(p, this->get_name());
  }
  return true;
}

#define PROTO(T)                     \
  template class data_type_weights<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
