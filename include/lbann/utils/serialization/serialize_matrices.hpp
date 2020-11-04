#pragma once
#ifndef LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_HPP_
#define LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_HPP_

#include "cereal_utils.hpp"
#include <lbann/utils/exception.hpp>

#include <El.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/polymorphic.hpp>

#include <stdexcept>

namespace cereal
{

/** @brief Save a CPU matrix to a text-based archive.
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
template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void save(ArchiveT& archive, El::Matrix<T, El::Device::CPU> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  archive(make_nvp("height", mat.Height()),
          make_nvp("width", mat.Width()));
}

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
void save(ArchiveT& archive,
          El::Matrix<T, El::Device::CPU> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  archive(mat.Height(), mat.Width());
  if (mat.Contiguous())
  {
    archive(binary_data(mat.LockedBuffer(),
                        mat.LDim()*mat.Width()*sizeof(T)));
  }
  else
  {
    for (El::Int col = 0; col < mat.Width(); ++col)
      archive(binary_data(mat.LockedBuffer() + col*mat.LDim(),
                          mat.Height()*sizeof(T)));
  }
}

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
template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void load(ArchiveT& archive, El::Matrix<T, El::Device::CPU>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  El::Int height, width;
  archive(height, width);
  mat.Resize(height, width);
}

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
          El::Matrix<T, El::Device::CPU>& mat)
{
  LBANN_ASSERT(!mat.Viewing());
  El::Int height, width;
  archive(height, width);
  mat.Resize(height, width);
  archive(binary_data(mat.Buffer(),
                      mat.Height()*mat.Width()*sizeof(T)));
}

// DistMatrix

template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void save(ArchiveT& ar, El::AbstractDistMatrix<T> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());

  if (mat.DistRank() == mat.Root())
    ar(make_nvp("global_height", mat.Height()),
       make_nvp("global_width", mat.Width()));
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void save(ArchiveT& ar, El::AbstractDistMatrix<T> const& mat)
{
  LBANN_ASSERT(!mat.Viewing());

  using CircMatType =
    El::DistMatrix<T,El::CIRC,El::CIRC,El::ELEMENT,El::Device::CPU>;
  CircMatType circ_mat(mat);
  // Only the root writes to the archive.
  if (circ_mat.CrossRank() == circ_mat.Root())
    ar(mat.Height(), mat.Width(), circ_mat.LockedMatrix());
}

template <typename ArchiveT, typename T,
          lbann::utils::WhenTextArchive<ArchiveT> = 1>
void load(ArchiveT& ar, El::AbstractDistMatrix<T>& mat)
{
  El::Int height, width;
  if (mat.DistRank() == mat.Root())
    ar(height, width);
  El::mpi::Broadcast(height, mat.Root(), mat.DistComm(),
                     El::SyncInfo<El::Device::CPU>{});
  El::mpi::Broadcast(width, mat.Root(), mat.DistComm(),
                     El::SyncInfo<El::Device::CPU>{});
  mat.Resize(height, width);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// CIRC-CIRC matrix loading.
template <typename ArchiveT, typename T,
          lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
void load(
  ArchiveT& ar,
  El::DistMatrix<T,El::CIRC,El::CIRC,El::ELEMENT,El::Device::CPU>& cmat)
{
  El::Int height, width;
  // Only the root reads from the archive.
  if (cmat.CrossRank() == cmat.Root())
  {
    ar(height, width);
  }
  El::mpi::Broadcast(height, cmat.Root(), cmat.CrossComm(),
                     El::SyncInfo<El::Device::CPU>{});
  El::mpi::Broadcast(width, cmat.Root(), cmat.CrossComm(),
                     El::SyncInfo<El::Device::CPU>{});

  // Now make sure the height/width of the global matrix is right.
  cmat.Resize(height, width);
  if (cmat.CrossRank() == cmat.Root())
  {
    ar(cmat.Matrix());
  }
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Load a CPU DistMatrix from a non-text archive.
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
void load(ArchiveT& ar, El::AbstractDistMatrix<T>& mat)
{
  // Gather everything to root.
  El::DistMatrix<T,El::CIRC,El::CIRC,El::ELEMENT,El::Device::CPU> cmat(
    mat.Grid(), mat.Root());

  // Restore the CIRC-CIRC matrix.
  ar(cmat);

  // Get the data back into the "right" distribution.
  El::Copy(cmat, mat);
}

}// namespace cereal

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

}// namespace utils
}// namespace lbann

namespace cereal
{

template <typename DataT,
          El::Dist CDist,
          El::Dist RDist,
          El::DistWrap Wrap,
          El::Device D>
struct LoadAndConstruct<El::DistMatrix<DataT, CDist, RDist, Wrap, D>>
{
  using DistMatrixType = El::DistMatrix<DataT, CDist, RDist, Wrap, D>;
  using CircMatrixType = El::DistMatrix<DataT,
                                        El::CIRC, El::CIRC,
                                        Wrap,
                                        El::Device::CPU>;

  template <typename ArchiveT,
            lbann::utils::WhenTextArchive<ArchiveT> = 1>
  static void load_and_construct(
    ArchiveT & ar, cereal::construct<DistMatrixType> & construct)
  {
    El::Grid const& g = lbann::utils::get_current_grid();
    El::Int height, width;
    if (g.Rank() == 0)
      ar(height, width);
    El::mpi::Broadcast(height, /*root=*/0, g.Comm(),
                       El::SyncInfo<El::Device::CPU>{});
    El::mpi::Broadcast(width, /*root=*/0, g.Comm(),
                       El::SyncInfo<El::Device::CPU>{});
    construct(height, width,
              lbann::utils::get_current_grid(), /*root=*/0);
  }

  template <typename ArchiveT,
            lbann::utils::WhenNotTextArchive<ArchiveT> = 1>
  static void load_and_construct(
    ArchiveT & ar, cereal::construct<DistMatrixType> & construct)
  {
    // I think this should actually create a CircCirc matrix, but we'll see.
    CircMatrixType cmat(lbann::utils::get_current_grid(), /*root=*/0);
    ar(cmat);
    construct(cmat);
  }
};
}// namespace cereal

#endif // LBANN_UTILS_SERIALIZATION_SERIALIZE_MATRICES_HPP_
