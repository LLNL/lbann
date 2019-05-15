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

/// @todo Remove this file.

#ifndef LBANN_IO_H
#define LBANN_IO_H

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

namespace lbann {
/// @todo Deprecated.
int makedir(const char *dirname);
/// @todo Deprecated.
int exists(const char *filename);
/// @todo Deprecated.
int openread(const char *filename);
/// @todo Deprecated.
int closeread(int fd, const char *filename);
/// @todo Deprecated.
int openwrite(const char *filename);
/// @todo Deprecated.
int closewrite(int fd, const char *filename);
}

#endif // LBANN_IO_H
