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

#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstring>
#include <cstdio>

#define LBANN_PERSIST_INSTANTIATE
#include "lbann/io/persist.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/io/file_io.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "El.hpp"
#include "mpi.h"

/****************************************************
 * These functions will save a libElemental matrix
 * using a file-per-process
 ****************************************************/

/* TODO: user buffered I/O for more efficient writes */

/** Stores meta data needed to reconstruct matrix in memory after reading
 *  it back from a file */
struct layer_header {
  uint64_t rank;       /**< rank of MPI process that wrote the file */
  uint64_t width;      /**< global width of matrix */
  uint64_t height;     /**< global height of matrix */
  uint64_t localwidth; /**< local width of matrix on current process */
  uint64_t localheight;/**< local height of matrix on current process */
  uint64_t ldim;       /**< specifies padding of first dimension in local storage */
};

/** \brief Given an open file descriptor, file name, and a matrix, write the matrix
 *         to the file descriptor, return the number of bytes written */

template <typename TensorDataType>
bool lbann::persist::write_rank_distmat(persist_type type, const char *name, const El::AbstractDistMatrix<TensorDataType>& M) {
  // TODO: store in network order
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    LBANN_ERROR("invalid persist_type (", static_cast<int>(type), ")");
  }
  // skip all of this if matrix is not held on rank
  const El::Int localHeight = M.LocalHeight();
  const El::Int localWidth = M.LocalWidth();
  // If this is the case we will try to grab the matrix from model rank 0 on reload
  if(localHeight * localWidth == 0) { return true; }


  int fd = lbann::openwrite(filename.c_str());

  // build our header
  struct layer_header header;
  header.rank        = (uint64_t) M.Grid().Rank();
  header.width       = (uint64_t) M.Width();
  header.height      = (uint64_t) M.Height();
  header.localwidth  = (uint64_t) M.LocalWidth();
  header.localheight = (uint64_t) M.LocalHeight();
  header.ldim        = (uint64_t) M.LDim();

  // write the header to the file
  ssize_t write_rc = write(fd, &header, sizeof(header));
  if (write_rc != sizeof(header)) {
    // error!
  }
  m_bytes[type] += write_rc;

  // now write the data for our part of the distributed matrix
  const El::Int lDim = M.LDim();
  if(localHeight == lDim) {
    // the local dimension in memory matches the local height,
    // so we can write our data in a single shot
    auto *buf = (void *) M.LockedBuffer();
    El::Int bufsize = localHeight * localWidth * sizeof(DataType);
    write_rc = write(fd, buf, bufsize);
    if (write_rc != bufsize) {
      // error!
    }
    m_bytes[type] += write_rc;
  } else {
    // TODO: if this padding is small, may not be a big deal to write it out anyway
    // we've got some padding along the first dimension
    // while storing the matrix in memory, avoid writing the padding
    for(El::Int j = 0; j < localWidth; ++j) {
      auto *buf = (void *) M.LockedBuffer(0, j);
      El::Int bufsize = localHeight * sizeof(DataType);
      write_rc = write(fd, buf, bufsize);
      if (write_rc != bufsize) {
        // error!
      }
      m_bytes[type] += write_rc;
    }
  }
  return true;
}

/** \brief Given an open file descriptor, file name, and a matrix, read the matrix
 *         from the file descriptor, return the number of bytes read */
template <typename TensorDataType>
bool lbann::persist::read_rank_distmat(persist_type type, const char *name, El::AbstractDistMatrix<TensorDataType>& M) {
  // read in the header
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    LBANN_ERROR("invalid persist_type (", static_cast<int>(type), ")");
  }
  int fd = openread(filename.c_str());
  // file does not exist. we will try to grab matrix from rank 0
   if( fd == -1 ) {return false;}

  struct layer_header header;
  ssize_t read_rc = read(fd, &header, sizeof(header));
  if (read_rc != sizeof(header)) {
    LBANN_ERROR("failed to read layer header from file (attempted to read ",
                sizeof(header), " bytes from ", filename, ", but got ", read_rc, " bytes)");
  }
  m_bytes[type] += read_rc;

  // resize our global matrix
  El::Int height = header.height;
  El::Int width  = header.width;
  M.Resize(height, width);
  // TODO: check that header values match up
  const El::Int localheight = header.localheight;
  const El::Int localwidth = header.localwidth;
  if(M.ColStride() == 1 && M.RowStride() == 1) {
    if(M.Height() == M.LDim()) {
      auto *buf = (void *) M.Buffer();
      El::Int bufsize = localheight * localwidth * sizeof(DataType);
      read_rc = read(fd, buf, bufsize);
      if (read_rc != bufsize) {
        LBANN_ERROR("failed to read layer data from file (attempted to read ", bufsize,
                    " bytes from ", filename, ", but got ", read_rc, " bytes)");
      }
      m_bytes[type] += read_rc;
    } else {
      for(El::Int j = 0; j <  localwidth; ++j) {
        auto *buf = (void *) M.Buffer(0, j);
        El::Int bufsize = localheight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          LBANN_ERROR("failed to read layer data from file (attempted to read ",
                      bufsize, " bytes from ", filename, ", but got ", read_rc, " bytes)");
        }
        m_bytes[type] += read_rc;
      }
    }
  } else {
    const El::Int lDim = M.LDim();
    if(localheight == lDim) {
      auto *buf = (void *) M.Buffer();
      El::Int bufsize = localheight * localwidth * sizeof(DataType);
      read_rc = read(fd, buf, bufsize);
      if (read_rc != bufsize) {
        LBANN_ERROR("failed to read layer data from file (attempted to read ",
                    bufsize, " bytes from ", filename, ", but got ", read_rc, " bytes)");
      }
      m_bytes[type] += read_rc;
    } else {
      for(El::Int jLoc = 0; jLoc < localwidth; ++jLoc) {
        auto *buf = (void *) M.Buffer(0, jLoc);
        El::Int bufsize = localheight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          LBANN_ERROR("failed to read layer data from file (attempted to read ",
                      bufsize, " bytes from ", filename, ", but got ", read_rc, " bytes)");
        }
        m_bytes[type] += read_rc;
      }
    }
  }
  return true;
}

/****************************************************
 * Functions to read/write values to files
 ****************************************************/

lbann::persist::persist() {
  for(persist_type pt : persist_type_iterator()) {
    // initialize number of bytes written
    m_bytes[pt] = 0;
    // initialize file descriptors
    m_FDs[pt] = -1;
    m_filenames[pt] = "<unknown>";
  }
}

/** @todo BVE FIXME this should be refactored to only open the
    checkpoints files that we care about */
void lbann::persist::open_checkpoint(const std::string& dir) {
  // create directory for checkpoint
  lbann::makedir(dir.c_str());

  // copy checkpoint directory
  m_checkpoint_dir = dir;

  for(persist_type pt : persist_type_iterator()) {
    // open the file for writing
    m_filenames[pt] = dir + to_string(pt);
    // Do not explicitly open several files -- this state is saved via Cereal
    if(pt != persist_type::metrics &&
       pt != persist_type::testing &&
       pt != persist_type::validate &&
       pt != persist_type::testing_context &&
       pt != persist_type::training_context &&
       pt != persist_type::validation_context &&
       pt != persist_type::prediction_context) {
      m_FDs[pt] = lbann::openwrite(m_filenames[pt].c_str());
      if (m_FDs[pt] < 0) {
        LBANN_ERROR("failed to open file (", m_filenames[pt], ")");
      }
    }
  }
}

void lbann::persist::close_checkpoint() {
  for(persist_type pt : persist_type_iterator()) {
    if (m_FDs[pt] >= 0) {
      lbann::closewrite(m_FDs[pt], m_filenames[pt].c_str());
      m_FDs[pt] = -1;
      m_filenames[pt] = "<unknown>";
    }
  }
}

void lbann::persist::open_restart(const std::string& dir) {
  // copy checkpoint directory
  m_checkpoint_dir = dir;

  for(persist_type pt : persist_type_iterator()) {
    // open the file for reading
    m_filenames[pt] = dir + to_string(pt);
    if(pt != persist_type::metrics &&
       pt != persist_type::testing &&
       pt != persist_type::validate &&
       pt != persist_type::testing_context &&
       pt != persist_type::training_context &&
       pt != persist_type::validation_context &&
       pt != persist_type::prediction_context) {
      m_FDs[pt] = lbann::openread(m_filenames[pt].c_str());
      if (m_FDs[pt] < 0) {
        LBANN_ERROR("failed to open file (", m_filenames[pt], ")");
      }
    }
  }
}

void lbann::persist::close_restart() {
  for(persist_type pt : persist_type_iterator()) {
    if (m_FDs[pt] >= 0) {
      lbann::closeread(m_FDs[pt], m_filenames[pt].c_str());
      m_FDs[pt] = -1;
      m_filenames[pt] = "<unknown>";
    }
  }
}

template <typename TensorDataType>
bool lbann::persist::write_distmat(persist_type type, const char *name, El::AbstractDistMatrix<TensorDataType> *M) {
  // define full path to file to store matrix
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    LBANN_ERROR("invalid persist_type (", static_cast<int>(type), ")");
  }

  El::Write(*M, filename, El::BINARY, "");
  //Write_MPI(M, filename, BINARY, "");

  uint64_t bytes = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes[type] += bytes;

  return true;
}

template <typename TensorDataType>
bool lbann::persist::read_distmat(persist_type type, const char *name, El::AbstractDistMatrix<TensorDataType> *M) {
  // define full path to file to store matrix
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    LBANN_ERROR("invalid persist_type (", static_cast<int>(type), ")");
  }

  // check whether file exists
  int exists = lbann::exists(filename.c_str());
  if (! exists) {
    LBANN_ERROR("failed to read distributed matrix from file (", filename, ")");
    return false;
  }
  El::Read(*M, filename, El::BINARY, true);
  //Read_MPI(M, filename, BINARY, 1);

  uint64_t bytes = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes[type] += bytes;

  return true;
}

bool lbann::persist::write_bytes(persist_type type, const char *name, const void *buf, size_t size) {
  int fd = get_fd(type);
  std::string filename = get_filename(type);
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR("failed to write to fd ", fd,
                  " for file ", filename, " and field (", name, ")");
      return false;
    }
    m_bytes[type] += size;
  }
  return true;
}

bool lbann::persist::read_bytes(persist_type type, const char *name, void *buf, size_t size) {
  int fd = get_fd(type);
  std::string filename = get_filename(type);
  if (fd >= 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR("failed to read ", size, " bytes from fd ",
                  fd, " for file ", filename, " and field (", name,
                  ") at offset ", m_bytes[type]);
      return false;
    }
    m_bytes[type] += size;
  }
  else {
    return false;
  }
  return true;
}

bool lbann::persist::write_uint32(persist_type type, const char *name, uint32_t val) {
  return write_bytes(type, name, &val, sizeof(uint32_t));
}

bool lbann::persist::read_uint32(persist_type type, const char *name, uint32_t *val) {
  return read_bytes(type, name, val, sizeof(uint32_t));
}

bool lbann::persist::write_uint64(persist_type type, const char *name, uint64_t val) {
  return write_bytes(type, name, &val, sizeof(uint64_t));
}

bool lbann::persist::read_uint64(persist_type type, const char *name, uint64_t *val) {
  return read_bytes(type, name, val, sizeof(uint64_t));
}

bool lbann::persist::write_int32_contig(persist_type type, const char *name, const int32_t *buf, uint64_t count) {
  size_t bytes = count * sizeof(int32_t);
  return write_bytes(type, name, buf, bytes);
}

bool lbann::persist::read_int32_contig(persist_type type, const char *name, int32_t *buf, uint64_t count) {
  size_t bytes = count * sizeof(int32_t);
  return read_bytes(type, name, buf, bytes);
}

bool lbann::persist::write_float(persist_type type, const char *name, float val) {
  return write_bytes(type, name, &val, sizeof(float));
}

bool lbann::persist::read_float(persist_type type, const char *name, float *val) {
  return read_bytes(type, name, val, sizeof(float));
}

bool lbann::persist::write_double(persist_type type, const char *name, double val) {
  return write_bytes(type, name, &val, sizeof(double));
}

bool lbann::persist::read_double(persist_type type, const char *name, double *val) {
  return read_bytes(type, name, val, sizeof(double));
}

template <typename TensorDataType>
bool lbann::persist::write_datatype(persist_type type, const char *name, TensorDataType val) {
  return write_bytes(type, name, &val, sizeof(TensorDataType));
}

template <typename TensorDataType>
bool lbann::persist::read_datatype(persist_type type, const char *name, TensorDataType *val) {
  return read_bytes(type, name, val, sizeof(TensorDataType));
}

bool lbann::persist::write_string(persist_type type, const char *name, const char *val, int str_length) {
  return write_bytes(type, name, val, sizeof(char) * str_length);
}

bool lbann::persist::read_string(persist_type type, const char *name, char *val, int str_length) {
  return read_bytes(type, name, val, sizeof(char) * str_length);
}

int lbann::persist::get_fd(persist_type type) const {
  return m_FDs.at(type);
}

std::string lbann::persist::get_filename(persist_type type) const {
  return m_filenames.at(type);
}

/****************************************************
 * Functions to read/write values to files
 ****************************************************/

template <typename TensorDataType>
bool lbann::write_distmat(int fd, const char *name, DistMatDT<TensorDataType> *M, uint64_t *bytes) {
  El::Write(*M, name, El::BINARY, "");
  //Write_MPI(M, name, BINARY, "");

  uint64_t bytes_written = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(TensorDataType);
  *bytes += bytes_written;

  return true;
}

template <typename TensorDataType>
bool lbann::read_distmat(int fd, const char *name, DistMatDT<TensorDataType> *M, uint64_t *bytes) {
  // check whether file exists
  int exists = lbann::exists(name);
  if (! exists) {
    LBANN_ERROR("failed to read distributed matrix from file (", name, ")");
    return false;
  }

  El::Read(*M, name, El::BINARY, true);
  //Read_MPI(M, name, BINARY, 1);

  uint64_t bytes_read = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(TensorDataType);
  *bytes += bytes_read;

  return true;
}

bool lbann::write_bytes(int fd, const char *name, const void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR("failed to write file (", name, ")");
      return false;
    }
  }
  return true;
}

bool lbann::read_bytes(int fd, const char *name, void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR("failed to read file (", name, ")");
      return false;
    }
  }
  return true;
}

bool lbann::write_uint32(int fd, const char *name, uint32_t val) {
  return lbann::write_bytes(fd, name, &val, sizeof(uint32_t));
}

bool lbann::read_uint32(int fd, const char *name, uint32_t *val) {
  return lbann::read_bytes(fd, name, val, sizeof(uint32_t));
}

bool lbann::write_uint64(int fd, const char *name, uint64_t val) {
  return lbann::write_bytes(fd, name, &val, sizeof(uint64_t));
}

bool lbann::read_uint64(int fd, const char *name, uint64_t *val) {
  return lbann::read_bytes(fd, name, val, sizeof(uint64_t));
}

bool lbann::write_int32_contig(int fd, const char *name, const int32_t *buf, uint64_t count) {
  size_t bytes = count * sizeof(int32_t);
  return lbann::write_bytes(fd, name, buf, bytes);
}

bool lbann::read_int32_contig(int fd, const char *name, int32_t *buf, uint64_t count) {
  size_t bytes = count * sizeof(int32_t);
  return lbann::read_bytes(fd, name, buf, bytes);
}

bool lbann::write_float(int fd, const char *name, float val) {
  return lbann::write_bytes(fd, name, &val, sizeof(float));
}

bool lbann::read_float(int fd, const char *name, float *val) {
  return lbann::read_bytes(fd, name, val, sizeof(float));
}

bool lbann::write_double(int fd, const char *name, double val) {
  return lbann::write_bytes(fd, name, &val, sizeof(double));
}

bool lbann::read_double(int fd, const char *name, double *val) {
  return lbann::read_bytes(fd, name, val, sizeof(double));
}

bool lbann::write_string(int fd, const char *name, const char *buf, size_t size) {
  if (fd > 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR("failed to write file (", name, ")");
      return false;
    }
  }
  return true;
}

bool lbann::read_string(int fd, const char *name, char *buf, size_t size) {
  if (fd > 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc <= 0) {
      LBANN_ERROR("failed to read file (", name, ")");
      return false;
    }
  }
  return true;
}

namespace lbann {

#define PROTO(T)                     \
  template bool persist::write_rank_distmat<T>(                        \
    persist_type type, const char *name, const El::AbstractDistMatrix<T>& M); \
  template bool persist::read_rank_distmat<T>(                         \
    persist_type type, const char *name, El::AbstractDistMatrix<T>& M);       \
  template bool persist::write_distmat<T>(                             \
    persist_type type, const char *name, El::AbstractDistMatrix<T> *M);       \
  template bool persist::read_distmat<T>(                              \
    persist_type type, const char *name, El::AbstractDistMatrix<T> *M);       \
  template bool persist::write_datatype<T>(                            \
    persist_type type, const char *name, T val);                              \
  template bool persist::read_datatype<T>(                             \
    persist_type type, const char *name, T *val);                             \
  template bool write_distmat<T>(                                      \
    int fd, const char *name, DistMatDT<T> *M, uint64_t *bytes);              \
  template bool read_distmat<T>(                                       \
    int fd, const char *name, DistMatDT<T> *M, uint64_t *bytes)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
