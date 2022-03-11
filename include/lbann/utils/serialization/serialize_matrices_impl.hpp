////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_IMPL_HPP_
#define LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_IMPL_HPP_

#include "lbann/utils/serialization/serialize_matrices.hpp"

// These really belong in Elemental; let's just extend that.
namespace El
{

template <typename ArchiveT, typename T>
void save(ArchiveT& ar, ::El::AbstractMatrix<T> const& mat)
{
  switch (mat.GetDevice())
  {
  case ::El::Device::CPU:
    save(ar, static_cast<::El::Matrix<T,::El::Device::CPU> const&>(mat));
    break;
#ifdef LBANN_HAS_GPU
  case ::El::Device::GPU:
    save(ar, static_cast<::El::Matrix<T,::El::Device::GPU> const&>(mat));
    break;
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Unknown device.");
  }
}

template <typename ArchiveT, typename T, ::El::Device D,
          lbann::utils::WhenTextArchive<ArchiveT>>
void save(ArchiveT& ar, ::El::Matrix<T,D> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ar(::cereal::make_nvp("height", mat.Height()),
     ::cereal::make_nvp("width", mat.Width()));
}

namespace details
{
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void do_save(ArchiveT& ar,
             ::El::Matrix<T, ::El::Device::CPU> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ar(mat.Height(), mat.Width());
  if (mat.Contiguous())
  {
    ar(::cereal::binary_data(mat.LockedBuffer(),
                             mat.LDim()*mat.Width()*sizeof(T)));
  }
  else
  {
    for (::El::Int col = 0; col < mat.Width(); ++col)
      ar(::cereal::binary_data(mat.LockedBuffer() + col*mat.LDim(),
                               mat.Height()*sizeof(T)));
  }
}

#ifdef LBANN_HAS_GPU
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void do_save(ArchiveT& ar,
             ::El::Matrix<T, ::El::Device::GPU> const& mat)
{
  ::El::Matrix<T, ::El::Device::CPU> cpu_mat(mat);
  do_save(ar, cpu_mat);
}
#endif // LBANN_HAS_GPU
}// namespace details

/** @brief Save a matrix to a binary archive. */
template <typename ArchiveT, typename T, ::El::Device D,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void save(ArchiveT& ar, ::El::Matrix<T, D> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  details::do_save(ar, mat);
}

template <typename ArchiveT, typename T, ::El::Device D>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::Matrix<T,D> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  // Forward to the underlying archive on Root.
  ar.save_on_root(mat);
}


template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void load(ArchiveT& archive, ::El::AbstractMatrix<T>& mat)
{
  switch (mat.GetDevice())
  {
  case ::El::Device::CPU:
    load(archive, static_cast<::El::Matrix<T,::El::Device::CPU>&>(mat));
    break;
#ifdef LBANN_HAS_GPU
  case ::El::Device::GPU:
    load(archive, static_cast<::El::Matrix<T,::El::Device::GPU>&>(mat));
    break;
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Unknown device.");
  }
}

template <typename ArchiveT, typename T, ::El::Device D,
          lbann::utils::WhenTextArchive<ArchiveT>>
void load(ArchiveT& archive, ::El::Matrix<T, D>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ::El::Int height, width;
  archive(CEREAL_NVP(height), CEREAL_NVP(width));
  mat.Resize(height, width);
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void load(ArchiveT& archive,
          ::El::Matrix<T, ::El::Device::CPU>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ::El::Int height, width;
  archive(CEREAL_NVP(height), CEREAL_NVP(width));
  mat.Resize(height, width);
  archive(::cereal::binary_data(mat.Buffer(),
                                mat.Height()*mat.Width()*sizeof(T)));
}

#if defined LBANN_HAS_GPU
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void load(ArchiveT& archive,
          ::El::Matrix<T, ::El::Device::GPU>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ::El::Matrix<T, ::El::Device::CPU> cpu_mat;
  load(archive, cpu_mat);
  ::El::Copy(cpu_mat, mat);
}
#endif // defined LBANN_HAS_GPU

template <typename ArchiveT, typename T, ::El::Device D>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::Matrix<T,D>& mat)
{
  LBANN_ASSERT(!mat.Viewing());

  // Restore the local matrix, then handle the Bcast
  ar.load_on_root(mat);

  // First broadcast the size information.
  auto height = mat.Height();
  auto width = mat.Width();
  ::El::mpi::Broadcast(height, ar.root(), ar.grid().Comm(),
                     ::El::SyncInfo<::El::Device::CPU>{});
  ::El::mpi::Broadcast(width, ar.root(), ar.grid().Comm(),
                     ::El::SyncInfo<::El::Device::CPU>{});
  // Resize _should_ be a no-op if the size doesn't change, but I'm
  // not actually 100% sure.
  if (!ar.am_root())
    mat.Resize(height, width);

  // Finally the matrix data.
  ::El::Broadcast(
    static_cast<::El::AbstractMatrix<T>&>(mat),
    ar.grid().Comm(), ar.root());
}

// DistMatrix

template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT>>
void save(ArchiveT& ar, ::El::AbstractDistMatrix<T> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ar(::cereal::make_nvp("global_height", mat.Height()),
     ::cereal::make_nvp("global_width", mat.Width()));
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT>>
void load(ArchiveT& ar, ::El::AbstractDistMatrix<T>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ::El::Int global_height, global_width;
  ar(::cereal::make_nvp("global_height", global_height),
     ::cereal::make_nvp("global_width", global_width));
  mat.Resize(global_height, global_width);
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void save(ArchiveT& ar, ::El::AbstractDistMatrix<T> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  // Binary archives don't use NVPs, so there's no point in making
  // them here.
  ar(mat.Height(),
     mat.Width(),
     mat.LockedMatrix());
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void load(ArchiveT& ar, ::El::AbstractDistMatrix<T>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  ::El::Int global_height, global_width;
  ar(global_height, global_width);
  mat.Resize(global_height, global_width);
#ifdef LBANN_DEBUG
  ::El::Matrix<T, ::El::Device::CPU> mat_cpu;
  ar(mat_cpu);
  LBANN_ASSERT(mat_cpu.Height() == mat.LocalHeight());
  LBANN_ASSERT(mat_cpu.Width() == mat.LocalWidth());
  mat.Matrix() = mat_cpu;
#else
  ar(mat.Matrix());
#endif
}


template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT>>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T> const& mat)
{
  ar(::cereal::make_nvp("global_height", mat.Height()),
     ::cereal::make_nvp("global_width",  mat.Width()));
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT>>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T>& mat)
{
  El::Int height, width;
  ar(::cereal::make_nvp("global_height", height),
     ::cereal::make_nvp("global_width",  width));
  mat.Resize(height, width);
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  using CircMatType =
    ::El::DistMatrix<T,::El::CIRC,::El::CIRC,::El::ELEMENT,::El::Device::CPU>;
  CircMatType circ_mat(mat);
  CircMatType circ_mat_ar(ar.grid(), ar.root());
  if (circ_mat.DistData() == circ_mat_ar.DistData()) {
    circ_mat_ar = std::move(circ_mat);
  }
  else {
    ::El::copy::Translate(circ_mat, circ_mat_ar);
  }
  save(ar, circ_mat_ar);
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::AbstractDistMatrix<T>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  using CircMatType =
    ::El::DistMatrix<T,::El::CIRC,::El::CIRC,::El::ELEMENT,::El::Device::CPU>;

  // Do the root process read.
  CircMatType circ_mat(mat.Grid(), mat.Root());
  CircMatType circ_mat_ar(ar.grid(), ar.root());
  load(ar, circ_mat_ar);
  if (circ_mat.DistData() == circ_mat_ar.DistData()) {
    circ_mat = std::move(circ_mat_ar);
  }
  else {
    ::El::copy::Translate(circ_mat_ar, circ_mat);
  }

  // Distribute the data
  ::El::Copy(circ_mat, mat);
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void save(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
          ::El::DistMatrix<T,::El::CIRC,::El::CIRC> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  LBANN_ASSERT(mat.Grid() == ar.grid());
  LBANN_ASSERT(mat.Root() == ar.root());
  ar(::cereal::make_nvp("global_height", mat.Height()),
     ::cereal::make_nvp("global_width", mat.Width()));
  save(ar, ::cereal::make_nvp("matrix_data", mat.LockedMatrix()));
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT>>
void load(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
          ::El::DistMatrix<T,::El::CIRC,::El::CIRC>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  LBANN_ASSERT(mat.Grid() == ar.grid());
  LBANN_ASSERT(mat.Root() == ar.root());

  // Restore the height/width using the usual mechanism, but WAIT on
  // the matrix, since the local matrix of CIRC,CIRC matrix is not
  // Bcast.
  ::El::Int height, width;
  ar(::cereal::make_nvp("global_height", height),
     ::cereal::make_nvp("global_width", width));

  // Restore the matrix data on the root process.
  mat.Resize(height, width);
  ar.load_on_root(mat.Matrix());
}

}// namespace El


// Dealing with smart pointers and object construction

namespace cereal
{

template <typename DataT,
          ::El::Dist CDist,
          ::El::Dist RDist,
          ::El::DistWrap Wrap,
          ::El::Device D>
template <typename ArchiveT,
          ::h2::meta::EnableWhen<::lbann::utils::IsBuiltinArchive<ArchiveT>, int>>
void
LoadAndConstruct<::El::DistMatrix<DataT, CDist, RDist, Wrap, D>>
::load_and_construct(
  ArchiveT & ar, cereal::construct<DistMatrixType> & construct)
{
  // Construct the matrix on the right grid.
  ::El::Grid const& g = lbann::utils::get_current_grid();
  construct(g, /*root=*/0);

  // Use the regular load function to restore its state. NOTE: do
  // *not* use ArchiveT::operator() here because it trys to open a
  // new scope, which can cause a variety of errors depending on the
  // underlying archive type.
  load(ar, *construct.ptr());
}

template <typename DataT,
          ::El::Dist CDist,
          ::El::Dist RDist,
          ::El::DistWrap Wrap,
          ::El::Device D>
template <typename ArchiveT>
void
LoadAndConstruct<::El::DistMatrix<DataT, CDist, RDist, Wrap, D>>
::load_and_construct(
  lbann::RootedInputArchiveAdaptor<ArchiveT> & ar,
  cereal::construct<DistMatrixType> & construct)
{
  construct(ar.grid(), /*root=*/0);
  load(ar, *construct.ptr());
}

}// namespace cereal

#endif // LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_IMPL_HPP_
