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

#include "lbann/lbann_base.hpp"
#include "El.hpp"

namespace lbann
{
//    typedef struct lbann_persist_struct {
//      int fd;
//    } lbann_persist;

    
    bool writeDist(int fd, const char* filename, const DistMat& M, uint64_t* bytes);
    bool readDist(int fd, const char* filename, DistMat& M, uint64_t* bytes);

    bool write_distmat(int fd, const char* name, DistMat* M, uint64_t* bytes);
    bool read_distmat (int fd, const char* name, DistMat* M, uint64_t* bytes);

    void write_uint32(int fd, const char* name, uint32_t  val);
    void read_uint32 (int fd, const char* name, uint32_t* val);

    void write_uint64(int fd, const char* name, uint64_t  val);
    void read_uint64 (int fd, const char* name, uint64_t* val);

    void write_float(int fd,  const char* name, float     val);
    void read_float (int fd,  const char* name, float*    val);

    void write_double(int fd, const char* name, double    val);
    void read_double (int fd, const char* name, double*   val);
}

#endif // LBANN_PERSIST_H
