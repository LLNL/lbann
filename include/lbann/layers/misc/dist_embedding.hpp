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

#ifndef LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED
#define LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED
#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"

#if defined(LBANN_HAS_SHMEM) || defined(LBANN_HAS_NVSHMEM)
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/memory.hpp"
#include "lbann/weights/weights_helpers.hpp"

namespace lbann {

/** @brief Embedding layer with distributed weights
 *
 *  This is similar to the embedding layer, which takes integer
 *  indices and returns embedding vectors from a lookup table.
 *  However, the embedding vectors are distributed between processes
 *  and one-sided inter-process communication is performed with
 *  OpenSHMEM (on CPU) or NVSHMEM (on GPU).
 *
 *  The main benefit of this model-parallel approach is to handle
 *  cases where the embedding vectors don't fit on one process. It
 *  should also have better scaling properties when the mini-batch
 *  size is very large.
 *
 *  To take advantage of sparse gradients, the distributed embedding
 *  layer provides the option to bypass the optimizer (which currently
 *  only supports dense gradients) and perform sparse SGD directly on
 *  the embedding weights. If enabled, SGD occurs during the layers
 *  "update" phase (i.e. in the virtual update_compute function).
 *  Otherwise, the layer converts sparse gradients to a dense tensor
 *  and passes it into the usual optimizer. This is a hack and will be
 *  deprecated once the optimizer class supports sparse gradients.
 *
 *  @warning This is experimental.
 *
 *  @todo Sparse SGD with optimizer class
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class dist_embedding_layer : public data_type_layer<TensorDataType>
{
  static_assert(
    Layout == data_layout::DATA_PARALLEL,
    "distributed embedding layer only supports data parallel layout");

public:
  dist_embedding_layer(size_t num_embeddings,
                       size_t embedding_dim,
                       bool sparse_sgd,
                       DataType learning_rate,
                       bool barrier_in_forward_prop);

  dist_embedding_layer(const dist_embedding_layer& other);
  dist_embedding_layer& operator=(const dist_embedding_layer& other);
  ~dist_embedding_layer();

  dist_embedding_layer* copy() const override;

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  description get_description() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  dist_embedding_layer();

  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;
  bool update_compute() override;

public:
  /** Metadata for an embedding vector from a remote process.
   *
   *  This should be treated as an internal implementation detail. It
   *  is only in public scope so it is available to CUDA kernels in an
   *  anonymous namespace.
   */
  struct vector_metadata
  {
    size_t source_rank{0};
    size_t source_index{0};
    size_t target_rank{0};
    size_t target_index{0};
    bool is_active{false};
  };

private:
  using LocalMat = El::Matrix<TensorDataType, Device>;

  /** @brief Non-blocking barrier
   *  @todo Handle case with non-default CUDA stream.
   *  @todo Move to comm header.
   */
  static void
  nb_barrier(lbann_comm& comm, const El::mpi::Comm& c, Al::request& req);

  void attach_embeddings_to_shmem_buffer();
  void apply_sparse_sgd_step(size_t num_gradients, LocalMat& local_embeddings);

  /** SHMEM buffer for embedding vectors.
   *
   *  If the embedding weights matrix is not already attached to a
   *  SHMEM buffer, then this layer allocates a SHMEM buffer and
   *  attaches it. In this case, the layer is responsible for managing
   *  the buffer.
   */
  TensorDataType* m_embeddings_buffer{nullptr};
  /** Allocated size of @c m_embeddings_buffer. */
  size_t m_embeddings_buffer_size{0};

  /** SHMEM buffer to communicate embedding vectors. */
  TensorDataType* m_workspace_buffer{nullptr};
  /** Allocated size of @c m_workspace_buffer. */
  size_t m_workspace_buffer_size{0};

  /** SHMEM buffer to communicate metadata for embedding vectors. */
  vector_metadata* m_metadata_buffer{nullptr};
  /** Allocated size of @c m_metadata_buffer. */
  size_t m_metadata_buffer_size{0};

  /** Request to synchronize non-blocking barriers.
   *
   *  Careful synchronization is required to ensure the correctness of
   *  asynchronous, one-sided communication via SHMEM buffers. After
   *  any modification to a SHMEM buffer (local or remote), a
   *  non-blocking barrier is launched to signal that the local
   *  process has finished its work. Before the next access to the
   *  SHMEM buffer, the non-blocking barrier is synchronized to make
   *  sure that all remote processes have finished their work and that
   *  the buffers are safe to access.
   */
  Al::request m_nb_barrier_request;

  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;

  /** Perform sparse SGD during backprop.
   *
   *  Bypasses optimizer class.
   */
  bool m_sparse_sgd;
  /** SGD learning rate. */
  DataType m_learning_rate;

  /** Perform a blocking barrier at the beginning of forward prop.
   *
   *  This layer performs synchronization with non-blocking barriers
   *  to ensure the correctness of asynchronous communication.
   *  However, gradient checking changes the embedding values without
   *  performing any synchronization. The quickest fix is to do a
   *  blocking barrier at the beginning of forward prop to make sure
   *  that all the embeddings are ready to be accessed.
   *
   *  @todo Think of a way to avoid this synchronization.
   */
  bool m_barrier_in_forward_prop;
};

// ---------------------------------------------
// Implementation
// ---------------------------------------------

template <typename T, data_layout L, El::Device D>
void dist_embedding_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_dist_embedding();
  msg->set_num_embeddings(m_num_embeddings);
  msg->set_embedding_dim(m_embedding_dim);
  msg->set_sparse_sgd(m_sparse_sgd);
  msg->set_learning_rate(m_learning_rate);
  msg->set_barrier_in_forward_prop(m_barrier_in_forward_prop);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType, Layout, Device>::dist_embedding_layer(
  size_t num_embeddings,
  size_t embedding_dim,
  bool sparse_sgd,
  DataType learning_rate,
  bool barrier_in_forward_prop)
  : data_type_layer<TensorDataType>(nullptr),
    m_num_embeddings{num_embeddings},
    m_embedding_dim{embedding_dim},
    m_sparse_sgd{sparse_sgd},
    m_learning_rate{learning_rate},
    m_barrier_in_forward_prop{barrier_in_forward_prop}
{

  // Learning rate is only used for sparse SGD
  if (!m_sparse_sgd) {
    m_learning_rate = -1.0;
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType, Layout, Device>::dist_embedding_layer()
  : dist_embedding_layer(1, 1, false, El::To<DataType>(1), false)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType, Layout, Device>::dist_embedding_layer(
  const dist_embedding_layer& other)
  : data_type_layer<TensorDataType>(other)
{
  LBANN_ERROR("copy constructor is invalid for dist_embedding_layer");
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType, Layout, Device>&
dist_embedding_layer<TensorDataType, Layout, Device>::operator=(
  const dist_embedding_layer& other)
{
  LBANN_ERROR("copy assignment operator is invalid for dist_embedding_layer");
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType, Layout, Device>*
dist_embedding_layer<TensorDataType, Layout, Device>::copy() const
{
  return new dist_embedding_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
dist_embedding_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "distributed embedding";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
dist_embedding_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
dist_embedding_layer<TensorDataType, Layout, Device>::get_device_allocation()
  const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
dist_embedding_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Num embeddings", m_num_embeddings);
  desc.add("Embedding dim", m_embedding_dim);
  desc.add("Using sparse SGD", m_sparse_sgd);
  desc.add("SGD learning rate", m_learning_rate);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  auto dims = this->get_input_dims();
  dims.push_back(static_cast<int>(m_embedding_dim));
  this->set_output_dims(dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Synchronize non-blocking barrier
  // Note: Make sure SHMEM buffers are safe to reset.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Construct default weights if needed
  // Note: Randomly drawn from normal distribution with mean 0 and
  // standard deviation 1.
  if (!this->has_weights()) {
    auto w = std::make_shared<data_type_weights<TensorDataType>>(comm);
    auto init = std::make_unique<normal_initializer<TensorDataType>>(0, 1);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->add_weights(w);
    this->m_model->add_weights(std::move(w));
  }
  if (this->num_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",
                this->num_weights(),
                ")");
  }

  // Configure embedding weights
  auto& embeddings = this->get_weights(0);
  {
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::VC;
    embeddings.set_dims({m_embedding_dim}, {m_num_embeddings});
    embeddings.set_matrix_distribution(dist);
  }

  // Destroy embedding optimizer and create dummy weights
  // Note: This layer manually performs sparse SGD on embedding
  // weights during backprop, so the embedding optimizer isn't needed.
  // However, the layer must send gradients to some optimizer to
  // prevent the model from optimizing the layer out of compute graph
  // during backprop. We get around this by creating dummy weights
  // with no entries.
  if (m_sparse_sgd) {
    embeddings.set_optimizer(nullptr);
    auto w = std::make_shared<data_type_weights<TensorDataType>>(comm);
    auto opt = std::make_unique<sgd<TensorDataType>>(0.);
    w->set_name(this->get_name() + "_dummy_weights");
    w->set_optimizer(std::move(opt));
    w->set_dims(1);
    w->set_matrix_distribution(embeddings.get_matrix_distribution());
    w->setup();
    this->add_weights(w);
    this->m_model->add_weights(std::move(w));
  }

  // Setup embedding weights
  embeddings.setup();
  attach_embeddings_to_shmem_buffer();

  // Non-blocking barrier
  // Note: Embeddings have been initialized
  nb_barrier(comm, comm.get_trainer_comm(), m_nb_barrier_request);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool dist_embedding_layer<TensorDataType, Layout, Device>::update_compute()
{

  // Apply sparse SGD if needed
  if (m_sparse_sgd) {
    const size_t input_size = this->get_input_size();
    const size_t mini_batch_size = this->get_prev_activations().Width();
    using ValuesGetter = weights_details::SafeWeightsAccessor<TensorDataType>;
    auto& embeddings = ValuesGetter::mutable_values(this->get_weights(0));
    auto& local_embeddings = dynamic_cast<LocalMat&>(embeddings.Matrix());
    apply_sparse_sgd_step(input_size * mini_batch_size, local_embeddings);
  }

  // Non-blocking barrier
  // Note: Embeddings are up-to-date.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);
  nb_barrier(comm, comm.get_trainer_comm(), m_nb_barrier_request);

  return true;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType, Layout, Device>::nb_barrier(
  lbann_comm& comm,
  const El::mpi::Comm& c,
  Al::request& req)
{
  static El::Matrix<float, Device> buffer;
  buffer.SetMemoryMode(0); // Don't use memory pool
  buffer.Resize(1, 1);
  comm.nb_allreduce(buffer, c, req);
}

// ---------------------------------------------
// Explicit template instantiation
// ---------------------------------------------

#ifdef LBANN_HAS_SHMEM
extern template class dist_embedding_layer<float,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::CPU>;
#endif // LBANN_HAS_SHMEM
#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_NVSHMEM)
extern template class dist_embedding_layer<float,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::GPU>;
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_HAS_NVSHMEM)

} // namespace lbann
#endif // defined(LBANN_HAS_SHMEM) || defined(LBANN_HAS_NVSHMEM)

// ---------------------------------------------
// Builder function
// ---------------------------------------------

namespace lbann {} // namespace lbann

#endif // LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED
