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

/** Get string describing tensor dimensions.
 *  The tensor is stored in a matrix, although there may be multiple
 *  dimensions corresponding to the matrix height and width.
 */
std::string get_dims_string(const std::vector<int>& matrix_height_dims,
                            const std::vector<int>& matrix_width_dims) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < matrix_height_dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << matrix_height_dims[i];
  }
  ss << ")x(";
  for (size_t i = 0; i < matrix_width_dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << matrix_width_dims[i];
  }
  ss << ")";
  return ss.str();
}

} // namespace

namespace lbann {

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights(lbann_comm* comm)
  : weights(comm) {}

template <typename TensorDataType>
data_type_weights<TensorDataType>::data_type_weights(const data_type_weights<TensorDataType>& other)
  : weights(other) {

  // Deep copies
  m_values.reset(other.m_values ? other.m_values->Copy() : nullptr);
  m_initializer.reset(other.m_initializer ?
                      other.m_initializer->copy() : nullptr);
  m_optimizer.reset(/*other.m_optimizer ?
                      other.m_optimizer->copy() : */nullptr);
  if (m_optimizer != nullptr) {
    m_optimizer->set_weights(this);
  }

}

template <typename TensorDataType>
data_type_weights<TensorDataType>& data_type_weights<TensorDataType>::operator=(const data_type_weights<TensorDataType>& other) {
  weights::operator=(other);

  // Deep copies
  m_values.reset(other.m_values ? other.m_values->Copy() : nullptr);
  m_initializer.reset(other.m_initializer ?
                      other.m_initializer->copy() : nullptr);
  m_optimizer.reset(/*other.m_optimizer ?
                      other.m_optimizer->copy() :*/ nullptr);
  if (m_optimizer != nullptr) {
    m_optimizer->set_weights(this);
  }

  return *this;
}

template <typename TensorDataType>
description data_type_weights<TensorDataType>::get_description() const {
  description desc = weights::get_description();

  // Optimizer
  if (m_optimizer != nullptr) {
    desc.add(m_optimizer->get_description());
  }

  // Initializer
  if (m_initializer != nullptr) {
    desc.add(m_initializer->get_description());
  }

  return desc;
}

// -----------------------------------------------
// Dimension accessors
// -----------------------------------------------
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_dims(std::vector<int> matrix_height_dims,
                                                 std::vector<int> matrix_width_dims) {
  weights::set_dims(matrix_height_dims, matrix_width_dims);
  if (m_values != nullptr) {
    const auto& height = get_matrix_height();
    const auto& width = get_matrix_width();
    if (m_values->Height() != height || m_values->Width() != width) {
      std::stringstream err;
      err << "attempted to set weights \"" << get_name() << "\" "
          << "with dimensions "
          << get_dims_string(matrix_height_dims, matrix_width_dims) << ", "
          << "but it is already setup with a "
          << m_values->Height() << " x " << m_values->Width() << " "
          << "weights matrix";
      LBANN_ERROR(err.str());
    }
  }
}

// -----------------------------------------------
// Initializer accessors
// -----------------------------------------------

template <typename TensorDataType>
weights_initializer<TensorDataType>* data_type_weights<TensorDataType>::get_initializer() {
  return const_cast<weights_initializer<TensorDataType>*>(static_cast<const data_type_weights&>(*this).get_initializer());
}
template <typename TensorDataType>
const weights_initializer<TensorDataType>* data_type_weights<TensorDataType>::get_initializer() const {
  return m_initializer.get();
}
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_initializer(std::unique_ptr<weights_initializer<TensorDataType>>&& init) {
  m_initializer = std::move(init);
}

// -----------------------------------------------
// Optimizer accessors
// -----------------------------------------------

template <typename TensorDataType>
data_type_optimizer<TensorDataType>* data_type_weights<TensorDataType>::get_optimizer() {
  return const_cast<data_type_optimizer<TensorDataType>*>(
           static_cast<const data_type_weights<TensorDataType>&>(*this).get_optimizer());
}
template <typename TensorDataType>
const data_type_optimizer<TensorDataType>* data_type_weights<TensorDataType>::get_optimizer() const {
  if (is_frozen()) {
    return nullptr;
  } else {
    return m_optimizer.get();
  }
}
template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_optimizer(std::unique_ptr<optimizer>&& opt) {
  m_optimizer = std::move(reinterpret_cast<std::unique_ptr<data_type_optimizer<TensorDataType>>&&>(opt));
}

// -----------------------------------------------
// Setup
// -----------------------------------------------

template <typename TensorDataType>
void data_type_weights<TensorDataType>::setup() {
  weights::setup();

  auto matrix_dist = get_matrix_distribution();
  // Construct weights matrix
  m_values.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(*matrix_dist.grid,
                                                                     matrix_dist.root,
                                                                     matrix_dist.colDist,
                                                                     matrix_dist.rowDist,
                                                                     (matrix_dist.blockHeight == 1
                                                                      && matrix_dist.blockWidth == 1 ?
                                                                      El::ELEMENT : El::BLOCK),
                                                                     matrix_dist.device));
  m_values->AlignWith(matrix_dist);
  m_values->Resize(get_matrix_height(), get_matrix_width());
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
El::AbstractDistMatrix<TensorDataType>& data_type_weights<TensorDataType>::get_values() {
  return const_cast<El::AbstractDistMatrix<TensorDataType>&>(static_cast<const data_type_weights&>(*this).get_values());
}
template <typename TensorDataType>
const El::AbstractDistMatrix<TensorDataType>& data_type_weights<TensorDataType>::get_values() const {
  if (m_values == nullptr) {
    LBANN_ERROR("attempted to access values of "
                "weights \"" + get_name() + "\" "
                "before they are setup");
  }
  return *m_values;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_values(const El::AbstractDistMatrix<TensorDataType>& values) {
  El::Copy(values, get_values());
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value, int index) {

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  const auto& size = get_size();
  if (index < 0 || index >= size) {
    std::stringstream err;
    err << "attempted to set value in "
        << "weights \"" << get_name() << "\""
        << "at index " << index << ", "
        << "but there are " << size << " values";
    LBANN_ERROR(err.str());
  }
#endif // LBANN_DEBUG

  // Set matrix entry
  const auto& height = get_matrix_height();
  set_value(value, index % height, index / height);

}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::set_value(TensorDataType value, std::vector<int> pos) {

  // Get tensor dimensions
  const auto& dims = get_dims();

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  bool valid = dims.size() == pos.size();
  for (size_t i = 0 ; i < dims.size(); ++i) {
    valid = valid && pos[i] >= 0 && pos[i] < dims[i];
  }
  if (!valid) {
    std::stringstream err;
    err << "attempted to set value in "
        << "weights \"" << get_name() << "\""
        << "at position (";
    for (size_t i = 0 ; i < pos.size(); ++i) {
      err << (i > 0 ? "x" : "") << pos[i];
    }
    err << ") in a tensor with dimensions ";
    for (size_t i = 0 ; i < dims.size(); ++i) {
      err << (i > 0 ? "x" : "") << dims[i];
    }
    LBANN_ERROR(err.str());
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
  const auto& height = get_matrix_height();
  const auto& width = get_matrix_width();
  if (row < 0 || row >= height || col < 0 || col > width ) {
    std::stringstream err;
    err << "attempted to set weights value "
        << "in weights \"" << get_name() << "\""
        << "at entry (" << row << "," << col << ") "
        << "in a " << height << "x" << width << " matrix";
    LBANN_ERROR(err.str());
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
    El::Scale(DataType(1) / values.RedundantSize(), values);
    get_comm().allreduce(values, values.RedundantComm());
  }
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::reconcile_values(Al::request& req) {
  auto& values = get_values();
  if (values.RedundantSize() > 1) {
    El::Scale(DataType(1) / values.RedundantSize(), values);
    get_comm().nb_allreduce(values, values.RedundantComm(), req);
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
  sprintf(l_name, "weights_%s_%lldx%lld", get_name().c_str(), m_values->Height(), m_values->Width());
  // write weights using persist call -- uses Elemental's write function.
  p.write_distmat(persist_type::model, l_name, m_values.get());
  // if saving training state, also write out state of optimizer
  if (m_optimizer != nullptr) {
    m_optimizer->save_to_checkpoint_shared(p, get_name());
  }

  return true;
}

template <typename TensorDataType>
void data_type_weights<TensorDataType>::write_proto(lbann_data::WeightsData* proto) const {

  // Set proto properties
  proto->Clear();
  proto->set_name(get_name());
  for (const auto& d : get_dims()) {
    proto->mutable_shape()->add_dim(d);
  }
  proto->set_height(get_matrix_height());
  proto->set_width(get_matrix_width());

  // Write weight values to prototext on world master process
  CircMat<El::Device::CPU> values = *m_values; /// @todo What if weights are on GPU?
  values.SetRoot(0); /// @todo What if world master is not process 0?
  if (get_comm().am_world_master()) {
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
  auto f_name = El::BuildString("weights_", get_name(), "_",
                                m_values->Height(), "x", m_values->Width(),
                                ".bin");
  p.read_distmat(persist_type::model, f_name.c_str(), m_values.get());
  if (m_optimizer != nullptr) {
    m_optimizer->load_from_checkpoint_shared(p, get_name());
  }

  return true;
}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_save(std::string const& ckpt_dir,
                                                       std::vector<std::string> const& weight_list){
  // create weight file name to match to weight list entry
  auto l_name = El::BuildString("model_weights_", get_name(), "_",
                                m_values->Height(), "x", m_values->Width(), ".bin");
  auto it = std::find(weight_list.begin(),weight_list.end(),l_name);
  // If match is found read in weight values.
  if(it != weight_list.end()) {
    std::string full_path = ckpt_dir + *it;
    if(get_comm().am_world_master()) {
      std::cout << "Loading " << get_name() << " <- " << *it << "\n";
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
  auto l_name = El::BuildString("weights_", get_name(),
                                "_", m_values->LocalHeight(),
                                "x", m_values->LocalWidth(), ".bin");
  p.write_rank_distmat(persist_type::model, l_name.c_str(), *m_values);
  if (m_optimizer != nullptr) {
    m_optimizer->save_to_checkpoint_distributed(p, get_name());
  }
  return true;
}

template <typename TensorDataType>
bool data_type_weights<TensorDataType>::load_from_checkpoint_distributed(lbann::persist& p){
  // Functions identically to shared checkpoint except weights and parameters are loaded on a per rank basis
  auto l_name = El::BuildString("weights_", get_name(),
                                "_", m_values->LocalHeight(),
                                "x", m_values->LocalWidth(), ".bin");
  p.read_rank_distmat(persist_type::model, l_name.c_str(), *m_values);
  if (m_optimizer != nullptr) {
    m_optimizer->load_from_checkpoint_distributed(p, get_name());
  }
  return true;
}

template class data_type_weights<float>;

}  // namespace lbann
