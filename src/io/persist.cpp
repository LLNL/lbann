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
    throw lbann_exception("persist: invalid persist_type");
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
  // read in the header
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    throw lbann_exception("persist: invalid persist_type");
  }
  int fd = openread(filename.c_str());
  // file does not exist. we will try to grab matrix from rank 0
   if( fd == -1 ) {return false;}
 
  struct layer_header header;
  ssize_t read_rc = read(fd, &header, sizeof(header));
  if (read_rc != sizeof(header)) {
    // error!
    throw lbann_exception("Failed to read layer header");
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
        // error!
        throw lbann_exception("Failed to read layer data");
      }
      m_bytes += read_rc;
    } else {
      for(El::Int j = 0; j <  localwidth; ++j) {
        auto *buf = (void *) M.Buffer(0, j);
        El::Int bufsize = localheight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          // error!
          throw lbann_exception("Failed to read layer data");
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
        // error!
        throw lbann_exception("Failed to read layer data");
      }
      m_bytes += read_rc;
    } else {
      for(El::Int jLoc = 0; jLoc < localwidth; ++jLoc) {
        auto *buf = (void *) M.Buffer(0, jLoc);
        El::Int bufsize = localheight * sizeof(DataType);
        read_rc = read(fd, buf, bufsize);
        if (read_rc != bufsize) {
          // error!
          throw lbann_exception("Failed to read layer data");
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

  if(ckpt_type != callback_type::validation){
    #ifdef LBANN_HAS_HDF5
    sprintf(m_model_filename, "%s/model.h5", dir);
    checkpoint_file = new H5::H5File(m_model_filename, H5F_ACC_TRUNC);
    #else
    sprintf(m_model_filename, "%s/model", dir);
    m_model_fd = lbann::openwrite(m_model_filename);
    if (m_model_fd < 0) {
      throw lbann_exception(std::string("Failed to open file: ") + m_model_filename);
    } 

    sprintf(m_train_filename, "%s/train", dir);
    m_train_fd = lbann::openwrite(m_train_filename);
    if (m_train_fd < 0) {
      throw lbann_exception(std::string("Failed to open file: ") + m_train_filename);
    }
    #endif
  } 
  if (ckpt_type == callback_type::validation || ckpt_type == callback_type::batch){
    #ifdef LBANN_HAS_HDF5
    sprintf(m_model_filename, "%s/model.h5", dir);
    checkpoint_file = new H5::H5File(m_model_filename, H5F_ACC_RDWR);
    #else
    sprintf(m_validate_filename, "%s/validate", dir);
    m_validate_fd = lbann::openwrite(m_validate_filename);
    if (m_validate_fd < 0) {
      throw lbann_exception(std::string("Failed to open file: ") + m_validate_filename);
    }
    #endif
  }
}

void lbann::persist::close_checkpoint() {
  // close model file
  #ifndef LBANN_HAS_HDF5
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
  #else
  checkpoint_file->close();
  #endif
}

void lbann::persist::open_restart(const char *dir) {
  // copy checkpoint directory
  strcpy(m_checkpoint_dir, dir);
  // open the file for writing
  #ifdef LBANN_HAS_HDF5
  sprintf(m_model_filename, "%s/model.h5", dir);
  checkpoint_file = new H5::H5File(m_model_filename, H5F_ACC_RDONLY);
  #else
  sprintf(m_model_filename, "%s/model", dir);
  
  // define filename for train state
  sprintf(m_train_filename, "%s/train", dir);
  // define filename for validate phase state
  sprintf(m_validate_filename, "%s/validate", dir);  
  
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
  m_validate_fd = lbann::openread(m_validate_filename);
  if (m_validate_fd < 0) {
    // restart failed, throw exception
      std::cout << "Failed to read " << m_validate_filename << " Not an error if validation percent = 0" << std::endl;
    //throw lbann_exception(std::string("Failed to read file: ") + m_validate_filename); 
  }
  #endif
}

void lbann::persist::close_restart() {
  // close model file
  #ifdef LBANN_HAS_HDF5
  checkpoint_file->close();
  #else
  lbann::closeread(m_model_fd, m_model_filename);
  m_model_fd = -1;
  // close training file
  lbann::closeread(m_train_fd, m_train_filename);
  m_train_fd = -1;
  // close validate file
  lbann::closeread(m_validate_fd, m_validate_filename);
  m_validate_fd = -1;
  #endif
}

bool lbann::persist::write_distmat(persist_type type, const char *name, AbsDistMat *M) {
  // define full path to file to store matrix
  std::string filename = m_checkpoint_dir;
  if (type == persist_type::train) {
    filename += std::string("/train_") + name;
  } else if (type == persist_type::model) {
    filename += std::string("/model_") + name;
  } else {
    throw lbann_exception("persist: invalid persist_type");
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
    throw lbann_exception("persist: invalid persist_type");
  }

  // check whether file exists
  int exists = lbann::exists(filename.c_str());
  if (! exists) {
    throw lbann_exception(std::string("Failed to read distmat: ") + filename);
    return false;
  }
  El::Read(*M, filename, El::BINARY, true);

  uint64_t bytes = 2 * sizeof(El::Int) + M->Height() * M->Width() * sizeof(DataType);
  m_bytes += bytes;

  return true;
}

#ifdef LBANN_HAS_HDF5
H5::Group lbann::persist::getGroup(std::string group_name){
  H5::Group abs_group;
  if(checkpoint_file->exists(group_name))
    abs_group = checkpoint_file->openGroup(group_name);
  else
    abs_group = checkpoint_file->createGroup(group_name);  
  return abs_group; 
}

bool lbann::persist::write_hdf5_distmat(std::string group_name, const char *name, AbsDistMat *M, lbann_comm *comm) {
  const hsize_t row_count = M->Height();
  const hsize_t col_count = M->Width();
  const hsize_t dims[2]= {row_count,col_count};
  H5::PredType hdf5_type = cpp_to_hdf5(M->Get(0,0));
  H5::DataSpace dataspace = H5::DataSpace(2, dims);
  H5::Group weight_group;
  if( M->ColStride() == 1 && M->RowStride() == 1 ){
    if (comm->am_model_master()){
      weight_group = getGroup(group_name); 
      H5::DataSet dataset = weight_group.createDataSet(name, hdf5_type, dataspace);
      dataset.write(M->LockedBuffer(), hdf5_type); 
    }
  } else {
    CircMat<El::Device::CPU> temp = *M;
    if (comm->am_world_master()){
      weight_group = getGroup(group_name);
      H5::DataSet dataset = weight_group.createDataSet(name, hdf5_type, dataspace);
      dataset.write(temp.LockedBuffer(), hdf5_type);
    }
  }  
  return true;
}

bool lbann::persist::read_hdf5_distmat(std::string group_name, const char *name, AbsDistMat *M, lbann_comm *comm) {
    CircMat<El::Device::CPU> temp(M->Grid());
    temp.Resize(M->Height(),M->Width());
    if (comm->am_world_master()){
      H5::Group weight_group = checkpoint_file->openGroup(group_name);
      H5::DataSet ds = weight_group.openDataSet(name);
      H5::DataSpace dataspace= ds.getSpace();
      ds.read(temp.Buffer(), H5::PredType::NATIVE_FLOAT, dataspace);     
      temp.Resize(M->Height(),M->Width());
    } 
    temp.MakeSizeConsistent();
    El::Copy(temp, *M);    
    return true;
}
#endif

bool lbann::persist::write_bytes(persist_type type, const char *name, const void *buf, size_t size) {
  int fd = get_fd(type);
  if (fd >= 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
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
    if (rc != (ssize_t) size) {
      throw lbann_exception(std::string("Failed to read: ") + name);
      return false;
    }
    m_bytes += size;
  }
  else {
    return false;
  } 
  return true;
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
    throw lbann_exception(std::string("Failed to read distmat: ") + name);
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
      throw lbann_exception(std::string("Failed to write: ") + name);
      return false;
    }
  }
  return true;
}

bool lbann::read_bytes(int fd, const char *name, void *buf, size_t size) {
  if (fd >= 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc != (ssize_t) size) {
      throw lbann_exception(std::string("Failed to read: ") + name);
      return false;
    }
  }
  return true;
}

bool lbann::write_int32_contig(int fd, const char *name, const int32_t *buf, uint64_t count) {
  size_t bytes = count * sizeof(int32_t);
  return lbann::write_bytes(fd, name, buf, bytes);
}

bool lbann::read_int32_contig(int fd, const char *name, int32_t *buf, uint64_t count) {
  size_t bytes = count * sizeof(int32_t);
  return lbann::read_bytes(fd, name, buf, bytes);
}

bool lbann::write_string(int fd, const char *name, const char *buf, size_t size) {
  if (fd > 0) {
    ssize_t rc = write(fd, buf, size);
    if (rc != (ssize_t) size) {
      throw lbann_exception(std::string("Failed to write: ") + name);
      return false;
    }
  }
  return true;
}

bool lbann::read_string(int fd, const char *name, char *buf, size_t size) {
  if (fd > 0) {
    ssize_t rc = read(fd, buf, size);
    if (rc <= 0) {
      throw lbann_exception(std::string("Failed to read: ") + name);
      return false;
    }
  }
  return true;
}
