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

#include "lbann/utils/exception.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"

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

bool lbann::persist::write_rank_distmat(persist_type type, const char *name, const AbsDistMat& M) {
  // TODO: store in network order
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    std::stringstream err;
    err << "invalid persist_type (" << static_cast<int>(type) << ")";
    LBANN_ERROR(err.str());
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
  m_bytes += write_rc;

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
    m_bytes += write_rc;
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
      m_bytes += write_rc;
    }
  }
  return true;
}

/** \brief Given an open file descriptor, file name, and a matrix, read the matrix
 *         from the file descriptor, return the number of bytes read */
bool lbann::persist::read_rank_distmat(persist_type type, const char *name, AbsDistMat& M) {
  std::stringstream err;

  // read in the header
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    err << "invalid persist_type (" << static_cast<int>(type) << ")";
    LBANN_ERROR(err.str());
  }
  int fd = openread(filename.c_str());
  // file does not exist. we will try to grab matrix from rank 0
   if( fd == -1 ) {return false;}

  struct layer_header header;
  ssize_t read_rc = read(fd, &header, sizeof(header));
  if (read_rc != sizeof(header)) {
    err << "failed to read layer header from file "
        << "(attempted to read " << sizeof(header) << " bytes "
        << "from " << filename << ", "
        << "but got " << read_rc << " bytes)";
    LBANN_ERROR(err.str());
  }
  m_bytes += read_rc;

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
        err << "failed to read layer data from file "
            << "(attempted to read " << bufsize << " bytes "
            << "from " << filename << ", "
            << "but got " << read_rc << " bytes)";
        LBANN_ERROR(err.str());
      }
      m_bytes += read_rc;
    } else {
      for(El::Int j = 0; j <  localwidth; ++j) {
        auto *buf = (void *) M.Buffer(0, j);
        El::Int bufsize = localheight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          err << "failed to read layer data from file "
              << "(attempted to read " << bufsize << " bytes "
              << "from " << filename << ", "
              << "but got " << read_rc << " bytes)";
          LBANN_ERROR(err.str());
        }
        m_bytes += read_rc;
      }
    }
  } else {
    const El::Int lDim = M.LDim();
    if(localheight == lDim) {
      auto *buf = (void *) M.Buffer();
      El::Int bufsize = localheight * localwidth * sizeof(DataType);
      read_rc = read(fd, buf, bufsize);
      if (read_rc != bufsize) {
        err << "failed to read layer data from file "
            << "(attempted to read " << bufsize << " bytes "
            << "from " << filename << ", "
            << "but got " << read_rc << " bytes)";
        LBANN_ERROR(err.str());
      }
      m_bytes += read_rc;
    } else {
      for(El::Int jLoc = 0; jLoc < localwidth; ++jLoc) {
        auto *buf = (void *) M.Buffer(0, jLoc);
        El::Int bufsize = localheight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          err << "failed to read layer data from file "
              << "(attempted to read " << bufsize << " bytes "
              << "from " << filename << ", "
              << "but got " << read_rc << " bytes)";
          LBANN_ERROR(err.str());
        }
        m_bytes += read_rc;
      }
    }
  }
  return true;
}

/****************************************************
 * Functions to read/write values to files
 ****************************************************/

lbann::persist::persist() {
  // initialize number of bytes written
  m_bytes = 0;

  // initialize file descriptors
  m_model_fd = -1;
  m_train_fd = -1;
  m_validate_fd = -1;
}

void lbann::persist::open_checkpoint(const char *dir) {
  // create directory for checkpoint
  lbann::makedir(dir);

  // copy checkpoint directory
  strcpy(m_checkpoint_dir, dir);

  // open the file for writing
  sprintf(m_model_filename, "%s/model", dir);

  // define filename for train state
  sprintf(m_train_filename, "%s/train", dir);

  if(ckpt_type != callback_type::validation && ckpt_type != callback_type::inference){
    m_model_fd = lbann::openwrite(m_model_filename);
    if (m_model_fd < 0) {
      LBANN_ERROR(std::string{}
                  + "failed to open file (" + m_model_filename + ")");
    }

    m_train_fd = lbann::openwrite(m_train_filename);
    if (m_train_fd < 0) {
      LBANN_ERROR(std::string{}
                  + "failed to open file (" + m_train_filename + ")");
    }
  }
  if (ckpt_type == callback_type::validation || ckpt_type == callback_type::batch){
    sprintf(m_validate_filename, "%s/validate", dir);
    m_validate_fd = lbann::openwrite(m_validate_filename);
    if (m_validate_fd < 0) {
      LBANN_ERROR(std::string{}
                  + "failed to open file (" + m_validate_filename + ")");
    }
  }
}

void lbann::persist::close_checkpoint() {
  // close model file
  if (m_model_fd >= 0) {
    lbann::closewrite(m_model_fd, m_model_filename);
    m_model_fd = -1;
  }

  // close training file
  if (m_train_fd >= 0) {
    lbann::closewrite(m_train_fd, m_train_filename);
    m_train_fd = -1;
  }
  if (m_validate_fd >= 0) {
    lbann::closewrite(m_validate_fd, m_validate_filename);
    m_validate_fd = -1;
  }
}

void lbann::persist::open_restart(const char *dir) {
  // copy checkpoint directory
  strcpy(m_checkpoint_dir, dir);
  // open the file for writing
  sprintf(m_model_filename, "%s/model", dir);

  // define filename for train state
  sprintf(m_train_filename, "%s/train", dir);
  // define filename for validate phase state
  sprintf(m_validate_filename, "%s/validate", dir);

  m_model_fd = lbann::openread(m_model_filename);
  if (m_model_fd < 0) {
    LBANN_ERROR(std::string{}
                + "failed to read file (" + m_model_filename + ")");
  }

  m_train_fd = lbann::openread(m_train_filename);
  if (m_train_fd < 0) {
    LBANN_ERROR(std::string{}
                + "failed to read file (" + m_train_filename + ")");
  }
  m_validate_fd = lbann::openread(m_validate_filename);
  if (m_validate_fd < 0) {
    LBANN_WARNING(std::string{}
                  + "failed to read file (" + m_validate_filename + "), "
                  + "which is not an error if validation percent = 0");
  }
}

void lbann::persist::close_restart() {
  // close model file
  lbann::closeread(m_model_fd, m_model_filename);
  m_model_fd = -1;
  // close training file
  lbann::closeread(m_train_fd, m_train_filename);
  m_train_fd = -1;
  // close validate file
  lbann::closeread(m_validate_fd, m_validate_filename);
  m_validate_fd = -1;

}

bool lbann::persist::write_distmat(persist_type type, const char *name, AbsDistMat *M) {
  // define full path to file to store matrix
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    std::stringstream err;
    err << "invalid persist_type (" << static_cast<int>(type) << ")";
    LBANN_ERROR(err.str());
  }

  El::Write(*M, filename, El::BINARY, "");
  //Write_MPI(M, filename, BINARY, "");

  uint64_t bytes = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes += bytes;

  return true;
}

bool lbann::persist::read_distmat(persist_type type, const char *name, AbsDistMat *M) {
  // define full path to file to store matrix
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    std::stringstream err;
    err << "invalid persist_type (" << static_cast<int>(type) << ")";
    LBANN_ERROR(err.str());
  }

  // check whether file exists
  int exists = lbann::exists(filename.c_str());
  if (! exists) {
    LBANN_ERROR("failed to read distributed matrix from file (" + filename + ")");
    return false;
  }
  El::Read(*M, filename, El::BINARY, true);
  //Read_MPI(M, filename, BINARY, 1);

  uint64_t bytes = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes += bytes;

  return true;
}

bool lbann::persist::write_bytes(persist_type type, const char *name, const void *buf, size_t size) {
  int fd = get_fd(type);
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR(std::string{} + "failed to write file (" + name + ")");
      return false;
    }
    m_bytes += size;
  }
  return true;
}

bool lbann::persist::read_bytes(persist_type type, const char *name, void *buf, size_t size) {
  int fd = get_fd(type);
  if (fd >= 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR(std::string{} + "failed to read file (" + name + ")");
      return false;
    }
    m_bytes += size;
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

bool lbann::persist::write_datatype(persist_type type, const char *name, DataType val) {
  return write_bytes(type, name, &val, sizeof(DataType));
}

bool lbann::persist::read_datatype(persist_type type, const char *name, DataType *val) {
  return read_bytes(type, name, val, sizeof(DataType));
}

bool lbann::persist::write_string(persist_type type, const char *name, const char *val, int str_length) {
  return write_bytes(type, name, val, sizeof(char) * str_length);
}

bool lbann::persist::read_string(persist_type type, const char *name, char *val, int str_length) {
  return read_bytes(type, name, val, sizeof(char) * str_length);
}

int lbann::persist::get_fd(persist_type type) const {
  int fd = -1;
  if (type == persist_type::train) {
    fd = m_train_fd;
  } else if (type == persist_type::model) {
    fd = m_model_fd;
  } else if (type == persist_type::validate) {
    fd = m_validate_fd;
  }
  return fd;
}

/****************************************************
 * Functions to read/write values to files
 ****************************************************/

bool lbann::write_distmat(int fd, const char *name, DistMat *M, uint64_t *bytes) {
  El::Write(*M, name, El::BINARY, "");
  //Write_MPI(M, name, BINARY, "");

  uint64_t bytes_written = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  *bytes += bytes_written;

  return true;
}

bool lbann::read_distmat(int fd, const char *name, DistMat *M, uint64_t *bytes) {
  // check whether file exists
  int exists = lbann::exists(name);
  if (! exists) {
    LBANN_ERROR(std::string{}
                + "failed to read distributed matrix from file "
                + "(" + name + ")");
    return false;
  }

  El::Read(*M, name, El::BINARY, true);
  //Read_MPI(M, name, BINARY, 1);

  uint64_t bytes_read = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  *bytes += bytes_read;

  return true;
}

bool lbann::write_bytes(int fd, const char *name, const void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR(std::string{} + "failed to write file (" + name + ")");
      return false;
    }
  }
  return true;
}

bool lbann::read_bytes(int fd, const char *name, void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc != (ssize_t) size) {
      LBANN_ERROR(std::string{} + "failed to read file (" + name + ")");
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
      LBANN_ERROR(std::string{} + "failed to write file (" + name + ")");
      return false;
    }
  }
  return true;
}

bool lbann::read_string(int fd, const char *name, char *buf, size_t size) {
  if (fd > 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc <= 0) {
      LBANN_ERROR(std::string{} + "failed to read file (" + name + ")");
      return false;
    }
  }
  return true;
}
