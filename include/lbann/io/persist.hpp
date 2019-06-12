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
//
// lbann_file_io .hpp .cpp - Input / output utilities
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_PERSIST_H
#define LBANN_PERSIST_H

#include "lbann/base.hpp"
#include "El.hpp"

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
  inference,
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

  bool write_uint32(persist_type type, const char *name, uint32_t  val);
  bool read_uint32 (persist_type type, const char *name, uint32_t *val);

  bool write_uint64(persist_type type, const char *name, uint64_t  val);
  bool read_uint64 (persist_type type, const char *name, uint64_t *val);

  bool write_int32_contig(persist_type type, const char *name, const int32_t *buf, uint64_t count);
  bool read_int32_contig (persist_type type, const char *name, int32_t *buf, uint64_t count);

  bool write_float(persist_type type, const char *name, float  val);
  bool read_float (persist_type type, const char *name, float *val);

  bool write_string(persist_type type, const char *name, const char *val, int str_length);
  bool read_string (persist_type type, const char *name, char *val, int str_length);

  bool write_double(persist_type type, const char *name, double  val);
  bool read_double (persist_type type, const char *name, double *val);

  bool write_datatype(persist_type type, const char *name, DataType  val);
  bool read_datatype (persist_type type, const char *name, DataType *val);

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
