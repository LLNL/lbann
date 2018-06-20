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
//
// lbann_file_io .hpp .cpp - Input / output utilities
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_PERSIST_H
#define LBANN_PERSIST_H

#include "lbann/base.hpp"
#include "El.hpp"
#ifdef LBANN_HAS_HDF5
#include "H5Cpp.h"
#endif
namespace lbann {

enum class persist_type {
  train, // data should be saved in file with train data
  model, // data should be saved in file with model data
  validate 
};

enum class callback_type {
  batch,
  epoch,
  validation,
  invalid
};

class persist {
 protected:
  uint64_t m_bytes;
  int m_model_fd;
  int m_train_fd;
  int m_validate_fd;
  char m_model_filename[1024];
  char m_train_filename[1024];
  char m_validate_filename[1024];
  callback_type ckpt_type; 
 public:
  char m_checkpoint_dir[1024];
  #ifdef LBANN_HAS_HDF5
  H5::H5File* checkpoint_file; 
  #endif
 public:
  persist();
  ~persist() {};

  callback_type get_cb_type() const {
    return ckpt_type;
  }

  void set_cb_type(callback_type type){
    ckpt_type = type;
  }

  void open_checkpoint(const char *dir);
  void close_checkpoint();

  void open_restart(const char *dir);
  void close_restart();

  uint64_t get_bytes() const {
    return m_bytes;
  }

  void reset_bytes() {
    m_bytes = 0;
  }

  bool write_rank_distmat(persist_type type, const char *name, const AbsDistMat& M);
  bool read_rank_distmat(persist_type type, const char *name, AbsDistMat& M);

  bool write_distmat(persist_type type, const char *name, AbsDistMat *M);
  bool read_distmat (persist_type type, const char *name, AbsDistMat *M);

  bool write_bytes(persist_type type, const char *name, const void *buf, size_t size);
  bool read_bytes(persist_type type, const char *name, void *buf, size_t size);
  
  template<typename T>
  bool write_parameter(persist_type type, const char *name, T val) {
    return write_bytes(type, name, &val, sizeof(T)); 
  }
  
  template<typename T>
  bool read_parameter(persist_type type, const char *name, T *val) {
    return read_bytes(type, name, val, sizeof(T));
  }


  template<typename T>
  bool write_parameter_vector(persist_type type, const char *name, std::vector<T> val, int array_size) {
    return write_bytes(type, name, val.data(), array_size * sizeof(T));
  }
  
  template<typename T>
  bool read_parameter_vector(persist_type type, const char *name, std::vector<T> &val, int array_size) {
    return read_bytes(type, name, val.data(), array_size * sizeof(T));
  }

  bool write_string(persist_type type, const char *name, const char *val, int str_length);
  bool read_string (persist_type type, const char *name, char *val, int str_length);

  
  #ifdef LBANN_HAS_HDF5   
  
  H5::PredType cpp_to_hdf5(int val) { return H5::PredType::NATIVE_INT; } 
  H5::PredType cpp_to_hdf5(double val) { return H5::PredType::NATIVE_DOUBLE; }
  H5::PredType cpp_to_hdf5(float val) { return H5::PredType::NATIVE_FLOAT; }
  H5::PredType cpp_to_hdf5(long val) { return H5::PredType::NATIVE_LONG; }
  
  template<typename T>
  bool write_hdf5_parameter(H5::Group group_name, const char *name, T *val) {
    H5::DataSpace dataspace = H5::DataSpace();
    H5::PredType hdf5_type = cpp_to_hdf5(*val);
    H5::Attribute attribute = group_name.createAttribute(name, hdf5_type, dataspace);
    attribute.write(hdf5_type, static_cast<void*>(val));
    return true;
  }


  template<typename T>
  bool read_hdf5_parameter(H5::Group group_name, const char *name, T *val) {
    H5::Attribute attr = group_name.openAttribute(name);
    H5::DataType type = attr.getDataType();
    attr.read(type,val);     
    return true;
  }

  template<typename T>
  bool write_hdf5_array(H5::Group group_name, const char *name, std::vector<T> val) {
    const hsize_t arr_size = val.size();
    H5::PredType hdf5_type = cpp_to_hdf5(val[0]);
    H5::DataSpace dataspace = H5::DataSpace(1,&arr_size);
    H5::DataSet dataset = group_name.createDataSet(name, hdf5_type, dataspace);
    dataset.write(val.data(), hdf5_type);
    return true;
  }
 
 
  template<typename T>
  bool read_hdf5_array(H5::Group group_name, const char *name, std::vector<T> &val) {
    H5::DataSet ds = group_name.openDataSet(name);
    H5::DataSpace dataspace= ds.getSpace();
    ds.read(val.data(), H5::PredType::NATIVE_INT, dataspace);
    return true;
  }
   
  bool write_hdf5_distmat(H5::Group group_name, const char *name, AbsDistMat *M);
  bool read_hdf5_distmat(H5::Group group_name, const char *name, AbsDistMat *M);
  #endif

 private:
  int get_fd(persist_type type) const;
};

bool write_distmat(int fd, const char *name, DistMat *M, uint64_t *bytes);
bool read_distmat (int fd, const char *name, DistMat *M, uint64_t *bytes);

bool write_bytes(int fd, const char *name, const void *buf, size_t size);
bool read_bytes(int fd, const char *name, void *buf, size_t size);

bool write_uint32(int fd, const char *name, uint32_t  val);
bool read_uint32 (int fd, const char *name, uint32_t *val);

bool write_uint64(int fd, const char *name, uint64_t  val);
bool read_uint64 (int fd, const char *name, uint64_t *val);

bool write_int32_contig(int fd, const char *name, const int32_t *buf, uint64_t count);
bool read_int32_contig (int fd, const char *name, int32_t *buf, uint64_t count);

bool write_float(int fd, const char *name, float  val);
bool read_float (int fd, const char *name, float *val);

bool write_double(int fd, const char *name, double  val);
bool read_double (int fd, const char *name, double *val);

bool write_string(int fd, const char *name, const char *buf, size_t size);
bool read_string(int fd, const char *name, char *buf, size_t size);

} // namespace lbann

#endif // LBANN_PERSIST_H
