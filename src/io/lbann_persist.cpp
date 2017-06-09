////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "lbann/utils/lbann_exception.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/io/lbann_persist.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "El.h"
#include "mpi.h"

using namespace std;
using namespace El;

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
bool lbann::writeDist(int fd, const char *filename, const DistMat& M, uint64_t *bytes) {
  // TODO: store in network order

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
  *bytes += write_rc;

  // now write the data for our part of the distributed matrix
  const Int localHeight = M.LocalHeight();
  const Int localWidth = M.LocalWidth();
  const Int lDim = M.LDim();
  if(localHeight == lDim) {
    // the local dimension in memory matches the local height,
    // so we can write our data in a single shot
    void *buf = (void *) M.LockedBuffer();
    size_t bufsize = localHeight * localWidth * sizeof(DataType);
    write_rc = write(fd, buf, bufsize);
    if (write_rc != bufsize) {
      // error!
    }
    *bytes += write_rc;
  } else {
    // TODO: if this padding is small, may not be a big deal to write it out anyway
    // we've got some padding along the first dimension
    // while storing the matrix in memory, avoid writing the padding
    for(Int j = 0; j < localWidth; ++j) {
      void *buf = (void *) M.LockedBuffer(0, j);
      size_t bufsize = localHeight * sizeof(DataType);
      write_rc = write(fd, buf, bufsize);
      if (write_rc != bufsize) {
        // error!
      }
      *bytes += write_rc;
    }
  }

  return true;
}

/** \brief Given an open file descriptor, file name, and a matrix, read the matrix
 *         from the file descriptor, return the number of bytes read */
bool lbann::readDist(int fd, const char *filename, DistMat& M, uint64_t *bytes) {
  // read in the header
  struct layer_header header;
  ssize_t read_rc = read(fd, &header, sizeof(header));
  if (read_rc != sizeof(header)) {
    // error!
    throw lbann_exception("Failed to read layer header");
  }
  *bytes += read_rc;

  // resize our global matrix
  Int height = header.height;
  Int width  = header.width;
  M.Resize(height, width);

  // TODO: check that header values match up

  if(M.ColStride() == 1 && M.RowStride() == 1) {
    if(M.Height() == M.LDim()) {
      void *buf = (void *) M.Buffer();
      size_t bufsize = height * width * sizeof(DataType);
      read_rc = read(fd, buf, bufsize);
      if (read_rc != bufsize) {
        // error!
        throw lbann_exception("Failed to read layer data");
      }
      *bytes += read_rc;
    } else {
      for(Int j = 0; j < width; ++j) {
        void *buf = (void *) M.Buffer(0, j);
        size_t bufsize = height * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          // error!
          throw lbann_exception("Failed to read layer data");
        }
        *bytes += read_rc;
      }
    }
  } else {
    const Int localHeight = M.LocalHeight();
    const Int localWidth = M.LocalWidth();
    const Int lDim = M.LDim();
    if(localHeight == lDim) {
      void *buf = (void *) M.Buffer();
      size_t bufsize = localHeight * localWidth * sizeof(DataType);
      read_rc = read(fd, buf, bufsize);
      if (read_rc != bufsize) {
        // error!
        throw lbann_exception("Failed to read layer data");
      }
      *bytes += read_rc;
    } else {
      for(Int jLoc = 0; jLoc < localWidth; ++jLoc) {
        void *buf = (void *) M.Buffer(0, jLoc);
        size_t bufsize = localHeight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          // error!
          throw lbann_exception("Failed to read layer data");
        }
        *bytes += read_rc;
      }
    }
  }
  return true;
}

/****************************************************
 * These functions are written in a style that should
 * make it easy to add to libElemental
 ****************************************************/

/** \brief Given a distributed matrix, commit the MPI datatypes needed for MPI I/O */
static void create_types(const El::DistMatrix<DataType>& M, MPI_Datatype *mattype, MPI_Datatype *viewtype) {
  // TODO: we could cache these datatypes on Matrix object

  // initialize return params to known values
  *mattype  = MPI_DATATYPE_NULL;
  *viewtype = MPI_DATATYPE_NULL;

  // TODO: use TypeMap<>() and templating to figure this out
  MPI_Datatype type = El::mpi::TypeMap<DataType>();

  // get global width and height of matrix
  Int global_width  = M.Width();
  Int global_height = M.Height();

  // get local width and height, plus leading dimension of local matrix
  Int W    = M.LocalWidth();
  Int H    = M.LocalHeight();
  Int LDim = M.LDim();

  // create a datatype to describe libelemental data in memory,
  // data is stored in column-major order with a local height of H
  // and a local width of W, also the leading dimension LDim >= H
  // so there may be holes in our local buffer between consecutive
  // columns which we need to account for

  // first we have H consecutive elements in a column
  MPI_Datatype tmptype;
  MPI_Type_contiguous(H, type, &tmptype);

  // then there may be some holes at then end of our column,
  // since LDim >= H
  MPI_Datatype coltype;
  MPI_Aint extent = LDim * sizeof(DataType);
  MPI_Type_create_resized(tmptype, 0, extent, &coltype);
  MPI_Type_free(&tmptype);

  // finally we have W such columns
  MPI_Type_contiguous(W, coltype, mattype);
  MPI_Type_free(&coltype);
  MPI_Type_commit(mattype);

  // create datatype to desribe fileview for a collective IO operation
  // we will store matrix in column-major order in the file

  // get width and height of the process grid
  int rank    = M.Grid().Rank();
  int ranks   = M.Grid().Size();
  int pheight = M.Grid().Height();
  int pwidth  = M.Grid().Width();

  // TODO: need to account for alignment if user has set this

  // create_darray expects processes to be in row-major order,
  // find our global rank in row-major order
  int prow = M.Grid().Row();
  int pcol = M.Grid().Col();
  int row_major_rank = prow * pwidth + pcol;

  int gsizes[2];
  gsizes[0] = global_height;
  gsizes[1] = global_width;
  int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
  // TODO: if using block sizes > 1, then change dargs below (BlockHeight, BlockWidth)
  int dargs[2] = {1, 1};
  int psizes[2];
  psizes[0] = pheight;
  psizes[1] = pwidth;
  MPI_Type_create_darray(ranks, row_major_rank, 2, gsizes, distribs, dargs, psizes, MPI_ORDER_FORTRAN, type, viewtype);
  MPI_Type_commit(viewtype);

  return;
}

/** \brief Write the given a distributed matrix to the specified file using MPI I/O */
static void Write_MPI(const El::DistMatrix<DataType>& M, std::string basename = "DistMatrix", El::FileFormat format = El::BINARY, std::string title = "") {
  // TODO: error out if format != BINARY

  // TODO: use TypeMap<>() and templating to figure this out
  MPI_Datatype type = El::mpi::TypeMap<DataType>();

  // define our file name
  string filename = basename + "." + FileExtension(BINARY);
  const char *path = filename.c_str();

  // get MPI communicator
  MPI_Comm comm = M.Grid().Comm().comm;

  // get our rank
  int rank = M.Grid().Rank();

  // first, delete the existing file
  if (rank == 0) {
    /*
    int unlink_rc = unlink(path);
    if (unlink_rc != 0) {
        fprintf(stderr, "Error deleting file `%s'\n", path);
        fflush(stderr);
    }
    */
    MPI_File_delete(path, MPI_INFO_NULL);
  }

  // get global width and height of matrix
  Int global_width  = M.Width();
  Int global_height = M.Height();

  // define datatypes to describe local buffer and view into file
  MPI_Datatype mattype, viewtype;
  create_types(M, &mattype, &viewtype);

  // define hints for creating the file (e.g., number of stripes on Lustre)
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "striping_factor", "10");
  //MPI_Info_set(info, "striping_factor", "80");
  // TODO: specify number of writers?

  // open the file
  MPI_File fh;
  MPI_Status status;
  char datarep[] = "native";
  int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
  MPI_File_open(comm, path, amode, info, &fh);

  // done with the info object
  MPI_Info_free(&info);

  // truncate file to 0 bytes
//    MPI_File_set_size(fh, 0);

  // set our view to write header (height and width as unsigned 32-bit ints)
  MPI_Offset disp = 0;
  MPI_File_set_view(fh, disp, MPI_UINT32_T, MPI_UINT32_T, datarep, MPI_INFO_NULL);
  if (rank == 0) {
    uint32_t dimensions[2];
    dimensions[0] = global_height;
    dimensions[1] = global_width;
    MPI_File_write_at(fh, 0, dimensions, 2, MPI_UINT32_T, &status);
  }
  disp += 2 * sizeof(uint32_t);

  // set view to write data
  MPI_File_set_view(fh, disp, type, viewtype, datarep, MPI_INFO_NULL);

  // write our portion of the matrix, since we set our view using create_darray,
  // all procs write at offset 0, the file view will take care of interleaving appropriately
  const char *buf = (const char *) M.LockedBuffer();
  MPI_File_write_at_all(fh, 0, buf, 1, mattype, &status);

  // close file
  MPI_File_close(&fh);

  // free our datatypes
  MPI_Type_free(&mattype);
  MPI_Type_free(&viewtype);

  return;
}

/** \brief Read the specified file and initialize a distributed matrix using MPI I/O */
static void Read_MPI(El::DistMatrix<DataType>& M, std::string filename, El::FileFormat format = El::BINARY, bool sequential = false) {
  // TODO: error out if format != BINARY

  // TODO: use TypeMap<>() and templating to figure this out
  MPI_Datatype type = El::mpi::TypeMap<DataType>();

  // define our file name
  const char *path = filename.c_str();

  // get MPI communicator
  MPI_Comm comm = M.Grid().Comm().comm;

  // get our rank
  int rank = M.Grid().Rank();

  // open the file
  MPI_File fh;
  MPI_Status status;
  char datarep[] = "native";
  int amode = MPI_MODE_RDONLY;
  int rc = MPI_File_open(comm, path, amode, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) {
    if (rank == 0) {
      cout << "Failed to open file `" << path << "'" << endl;
    }
    return;
  }

  // set displacement to beginning of file
  MPI_Offset disp = 0;

  // set our view to read header (height and width as unsigned 32-bit ints)
  uint32_t dimensions[2];
  MPI_File_set_view(fh, disp, MPI_UINT32_T, MPI_UINT32_T, datarep, MPI_INFO_NULL);
  if (rank == 0) {
    MPI_File_read_at(fh, 0, dimensions, 2, MPI_UINT32_T, &status);
  }
  disp += 2 * sizeof(uint32_t);

  // broadcast dimensions from rank 0
  MPI_Bcast(dimensions, 2, MPI_UINT32_T, 0, comm);

  // resize matrix to hold data
  Int global_height = dimensions[0];
  Int global_width  = dimensions[1];
  M.Resize(global_height, global_width);

  // now define datatypes to describe local buffer and view into file
  MPI_Datatype mattype, viewtype;
  create_types(M, &mattype, &viewtype);

  // set view to write data
  MPI_File_set_view(fh, disp, type, viewtype, datarep, MPI_INFO_NULL);

  // write our portion of the matrix, since we set our view using create_darray,
  // all procs write at offset 0, the file view will take care of interleaving appropriately
  char *buf = (char *) M.Buffer();
  MPI_File_read_at_all(fh, 0, buf, 1, mattype, &status);

  // close file
  MPI_File_close(&fh);

  // free our datatypes
  MPI_Type_free(&mattype);
  MPI_Type_free(&viewtype);

  return;
}

/****************************************************
 * Functions to read/write values to files
 ****************************************************/

lbann::persist::persist() {
  // lookup our MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

  // initialize number of bytes written
  m_bytes = 0;

  // initialize file descriptors
  m_model_fd = -1;
  m_train_fd = -1;
}

void lbann::persist::open_checkpoint(const char *dir) {
  // create directory for checkpoint
  lbann::makedir(dir);

  // copy checkpoint directory
  strcpy(m_checkpoint_dir, dir);

  // define filename for model state
  sprintf(m_model_filename, "%s/model", dir);

  // define filename for train state
  sprintf(m_train_filename, "%s/train", dir);

  // open the file for writing
  int fd = -1;
  if (m_rank == 0) {
    m_model_fd = lbann::openwrite(m_model_filename);
    if (m_model_fd < 0) {
      // failed to open checkpoint file
    }

    m_train_fd = lbann::openwrite(m_train_filename);
    if (m_train_fd < 0) {
      // failed to open checkpoint file
    }
  }
}

void lbann::persist::close_checkpoint(void) {
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
}

void lbann::persist::open_restart(const char *dir) {
  // copy checkpoint directory
  strcpy(m_checkpoint_dir, dir);

  // define filename for model state
  sprintf(m_model_filename, "%s/model", dir);

  // define filename for train state
  sprintf(m_train_filename, "%s/train", dir);

  // open the file for writing
  int fd = -1;
  if (m_rank == 0) {
    m_model_fd = lbann::openread(m_model_filename);
    if (m_model_fd < 0) {
      // restart failed, throw exception
      throw lbann_exception(std::string("Failed to read file: ") + m_model_filename);
    }

    m_train_fd = lbann::openread(m_train_filename);
    if (m_train_fd < 0) {
      // restart failed, throw exception
      throw lbann_exception(std::string("Failed to read file: ") + m_train_filename);
    }
  }
}

void lbann::persist::close_restart(void) {
  // close model file
  if (m_model_fd >= 0) {
    lbann::closeread(m_model_fd, m_model_filename);
    m_model_fd = -1;
  }

  // close training file
  if (m_train_fd >= 0) {
    lbann::closeread(m_train_fd, m_train_filename);
    m_train_fd = -1;
  }
}

bool lbann::persist::write_distmat(persist_type type, const char *name, DistMat *M) {
  // define full path to file to store matrix
  char filename[1024];
  if (type == persist_type::train) {
    snprintf(filename, sizeof(filename), "%s/train_%s", m_checkpoint_dir, name);
  } else if (type == persist_type::model) {
    snprintf(filename, sizeof(filename), "%s/model_%s", m_checkpoint_dir, name);
  }

  Write(*M, filename, BINARY, "");
  //Write_MPI(M, filename, BINARY, "");

  uint64_t bytes = 2 * sizeof(int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes += bytes;

  return true;
}

bool lbann::persist::read_distmat(persist_type type, const char *name, DistMat *M) {
  // define full path to file to store matrix
  char filename[1024];
  if (type == persist_type::train) {
    snprintf(filename, sizeof(filename), "%s/train_%s", m_checkpoint_dir, name);
  } else if (type == persist_type::model) {
    snprintf(filename, sizeof(filename), "%s/model_%s", m_checkpoint_dir, name);
  }

  // check whether file exists
  int exists = lbann::exists(filename);
  if (! exists) {
    throw lbann_exception(std::string("Failed to read distmat: ") + filename);
    return false;
  }

  Read(*M, filename, BINARY, 1);
  //Read_MPI(M, filename, BINARY, 1);

  uint64_t bytes = 2 * sizeof(int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes += bytes;

  return true;
}

bool lbann::persist::write_bytes(persist_type type, const char *name, void *buf, size_t size) {
  int fd = get_fd(type);
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != size) {
      throw lbann_exception(std::string("Failed to write: ") + name);
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
    if (rc != size) {
      throw lbann_exception(std::string("Failed to read: ") + name);
      return false;
    }
    m_bytes += size;
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

bool lbann::persist::write_int32_contig(persist_type type, const char *name, int32_t *buf, uint64_t count) {
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

int lbann::persist::get_fd(persist_type type) {
  int fd = -1;
  if (type == persist_type::train) {
    fd = m_train_fd;
  } else if (type == persist_type::model) {
    fd = m_model_fd;
  }
  return fd;
}

/****************************************************
 * Functions to read/write values to files
 ****************************************************/

bool lbann::write_distmat(int fd, const char *name, DistMat *M, uint64_t *bytes) {
  Write(*M, name, BINARY, "");
  //Write_MPI(M, name, BINARY, "");

  uint64_t bytes_written = 2 * sizeof(int) + M->Height() * M->Width() * sizeof(DataType);
  *bytes += bytes_written;

  return true;
}

bool lbann::read_distmat(int fd, const char *name, DistMat *M, uint64_t *bytes) {
  // check whether file exists
  int exists = lbann::exists(name);
  if (! exists) {
    throw lbann_exception(std::string("Failed to read distmat: ") + name);
    return false;
  }

  Read(*M, name, BINARY, 1);
  //Read_MPI(M, name, BINARY, 1);

  uint64_t bytes_read = 2 * sizeof(int) + M->Height() * M->Width() * sizeof(DataType);
  *bytes += bytes_read;

  return true;
}

bool lbann::write_bytes(int fd, const char *name, void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != size) {
      throw lbann_exception(std::string("Failed to write: ") + name);
      return false;
    }
  }
  return true;
}

bool lbann::read_bytes(int fd, const char *name, void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc != size) {
      throw lbann_exception(std::string("Failed to read: ") + name);
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

bool lbann::write_int32_contig(int fd, const char *name, int32_t *buf, uint64_t count) {
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
