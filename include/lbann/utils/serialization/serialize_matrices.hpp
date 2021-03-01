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
#pragma once
#ifndef LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_HPP_
#define LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_HPP_

#include "cereal_utils.hpp"
#include "rooted_archive_adaptor.hpp"

#include <lbann/utils/exception.hpp>

#include <El.hpp>
#include <stdexcept>

// These really belong in Elemental; let's just extend that.
namespace El
{

/** @brief Save a matrix to a text-based archive.
 *
 *  For these text-based archives (XML and JSON), this will just
 *  output the matrix metadata. Thus, a true deserialization will not
 *  be possible. Since these archive types are primarily intended for
 *  debugging, this should not be a problem. If it is, open an issue
 *  and it will be remedied.
 *
 *  @warning It is the caller's responsibility to ensure that the
 *           matrix to be serialized is actually a data-owning
 *           matrix. Serializing views is not supported, and, in the
 *           context of LBANN, should be unnecessary as the views will
 *           be reestablished when setup() is called on the
 *           deserialized objects.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive into which the matrix will be written.
 *  @param mat The matrix to serialize.
 *  @throws lbann::exception Thrown when the matrix is actually a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T>
void save(ArchiveT& ar, ::El::AbstractMatrix<T> const& mat);

template <typename ArchiveT, typename T, ::El::Device D,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void save(ArchiveT& ar, ::El::Matrix<T,D> const& mat);

namespace details
{
/** @brief Save a CPU matrix to a non-text-based archive.
 *
 *  @warning It is the caller's responsibility to ensure that the
 *           matrix to be serialized is actually a data-owning
 *           matrix. Serializing views is not supported, and, in the
 *           context of LBANN, should be unnecessary as the views will
 *           be reestablished when setup() is called on the
 *           deserialized objects.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive into which the matrix will be written.
 *  @param mat The matrix to serialize.
 *  @throws lbann::exception Thrown when the matrix is actually a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void do_save(ArchiveT& ar,
             ::El::Matrix<T, ::El::Device::CPU> const& mat);

#ifdef LBANN_HAS_GPU
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void do_save(ArchiveT& ar,
             ::El::Matrix<T, ::El::Device::GPU> const& mat);
#endif // LBANN_HAS_GPU
}// namespace details

/** @brief Save a matrix to a binary archive. */
template <typename ArchiveT, typename T, ::El::Device D,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void save(ArchiveT& ar, ::El::Matrix<T, D> const& mat);

// Special treatment for the rooted archive.
template <typename ArchiveT, typename T, ::El::Device D>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::Matrix<T,D> const& mat);


template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(ArchiveT& archive, ::El::AbstractMatrix<T>& mat);

/** @brief "Load" a CPU Matrix from a text-based archive.
 *
 *  The "text-based" ("human-readable") archives (XML and JSON) will
 *  only contain basic matrix metadata (height and width). Thus, a
 *  true deserialization will not be possible. Instead, the matrix
 *  restored by one of these archive types will have the proper size
 *  but it will not have meaningful data. Since these archive types
 *  are primarily intended for debugging, this should not be a
 *  problem. If it is, open an issue and it will be remedied.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive from which the matrix will be read.
 *  @param mat The target matrix to deserialize.
 *  @throws lbann::exception The input matrix is already setup as a view.
 *  @todo Perhaps it's better to throw an exception for these archives?
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T, ::El::Device D,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void load(ArchiveT& archive, ::El::Matrix<T, D>& mat);

/** @brief Load a CPU Matrix from a non-text archive.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive from which the matrix will be read.
 *  @param mat The target matrix to deserialize.
 *  @throws lbann::exception The input matrix is already setup as a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(ArchiveT& archive,
          ::El::Matrix<T, ::El::Device::CPU>& mat);

#if defined LBANN_HAS_GPU
/** @brief Load a GPU Matrix from a non-text archive.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive from which the matrix will be read.
 *  @param mat The target matrix to deserialize.
 *  @throws lbann::exception The input matrix is already setup as a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(ArchiveT& archive,
          ::El::Matrix<T, ::El::Device::GPU>& mat);
#endif // defined LBANN_HAS_GPU

template <typename ArchiveT, typename T, ::El::Device D>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::Matrix<T,D>& mat);

// DistMatrix

/** @brief Save a distributed matrix to a text-based archive.
 *
 *  For these text-based archives (XML and JSON), this will just
 *  output the matrix metadata. Thus, a true deserialization will not
 *  be possible. Since these archive types are primarily intended for
 *  debugging, this should not be a problem. If it is, open an issue
 *  and it will be remedied.
 *
 *  @warning It is the caller's responsibility to ensure that the
 *           matrix to be serialized is actually a data-owning
 *           matrix. Serializing views is not supported, and, in the
 *           context of LBANN, should be unnecessary as the views will
 *           be reestablished when setup() is called on the
 *           deserialized objects.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive into which the matrix will be written.
 *  @param mat The distributed matrix to serialize.
 *  @throws lbann::exception Thrown when the matrix is actually a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void save(ArchiveT& ar, ::El::AbstractDistMatrix<T> const& mat);

/** @brief Load a DistMatrix from a text-based archive.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive from which the matrix will be read.
 *  @param mat The target matrix to deserialize into.
 *  @throws lbann::exception The input matrix is already setup as a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void load(ArchiveT& ar, ::El::AbstractDistMatrix<T>& mat);

/** @brief Save a distributed matrix to a non-text (binary) archive.
 *
 *  In this case, the binary matrix data will be saved to the archive,
 *  as well as the global height/width.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.  @param archive
 *  The Cereal archive into which the matrix will be written.  @param
 *  mat The distributed matrix to serialize.  @throws lbann::exception
 *  Thrown when the matrix is actually a view.  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void save(ArchiveT& ar, ::El::AbstractDistMatrix<T> const& mat);

/** @brief Load a DistMatrix from a non-text archive.
 *
 *  @tparam ArchiveT (Inferred) The Cereal archive type to use.
 *  @tparam T (Inferred) The data type of the matrix.
 *  @param archive The Cereal archive from which the matrix will be read.
 *  @param mat The target matrix to deserialize into.
 *  @throws lbann::exception The input matrix is already setup as a view.
 *  @ingroup serialization
 */
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(ArchiveT& ar, ::El::AbstractDistMatrix<T>& mat);


template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T> const& mat);

template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T>& mat);

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T> const& mat);

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T>& mat);

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::DistMatrix<T,::El::CIRC,::El::CIRC> const& mat);

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::DistMatrix<T,::El::CIRC,::El::CIRC>& mat);

}// namespace El


// Dealing with smart pointers and object construction

namespace lbann
{
namespace utils
{

/** @brief RAII grid management.
 *
 *  Instantiate a new one of these at every distinct deserialization
 *  scope (trainer, subgraph, whatever). Ensure it's destroyed when
 *  deserialization at that scope is complete.
 */
struct grid_manager
{
  grid_manager(Grid const& g);
  ~grid_manager();
};

/** @brief Get the current grid being used for deserialization.
 *
 *  If not in a deserialization scope, this will return the default
 *  grid (and generate a warning from Hydrogen).
 */
Grid const& get_current_grid() noexcept;

lbann_comm& get_current_comm() noexcept;
}// namespace utils
}// namespace lbann

namespace cereal
{

/** @brief Construct DistMatrix object from Cereal archives. */
template <typename DataT,
          ::El::Dist CDist,
          ::El::Dist RDist,
          ::El::DistWrap Wrap,
          ::El::Device D>
struct LoadAndConstruct<::El::DistMatrix<DataT, CDist, RDist, Wrap, D>>
{
  using DistMatrixType = ::El::DistMatrix<DataT, CDist, RDist, Wrap, D>;
  using CircMatrixType = ::El::DistMatrix<DataT,
                                        ::El::CIRC, ::El::CIRC,
                                        Wrap,
                                        ::El::Device::CPU>;

  template <typename ArchiveT,
            ::h2::meta::EnableWhen<::lbann::utils::IsBuiltinArchive<ArchiveT>, int> = 0>
  static void load_and_construct(
    ArchiveT & ar, cereal::construct<DistMatrixType> & construct);

  template <typename ArchiveT>
  static void load_and_construct(
    lbann::RootedInputArchiveAdaptor<ArchiveT> & ar,
    cereal::construct<DistMatrixType> & construct);
};// struct LoadAndConstruct<::El::DistMatrix<DataT, CDist, RDist, Wrap, D>>
}// namespace cereal

#endif // LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_HPP_
