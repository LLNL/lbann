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

/// @todo Remove this file.

#include "lbann/io/file_io.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstring>
#include <cstdio>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <string>
#include <sstream>


/// @todo Deprecated.
int lbann::makedir(const char *dir) {
  std::string dir_(dir);
  file::make_directory(dir_);
  return 1;
}

/// @todo Deprecated.
int lbann::exists(const char *file) {
  std::string file_(file);
  return (file::file_exists(file_) ? 1 : 0);
}

/// @todo Deprecated.
int lbann::openread(const char *file) {
  // open the file for writing
  int fd = open(file, O_RDONLY);
  if (fd == -1) {
  }
  return fd;
}

/// @todo Deprecated.
int lbann::closeread(int fd, const char *file) {
  // close file
  int close_rc = close(fd);
  if (close_rc == -1) {
    fprintf(stderr, "ERROR: Failed to close file `%s' (%d: %s) @ %s:%d\n",
            file, errno, strerror(errno), __FILE__, __LINE__
           );
    fflush(stderr);
  }

  return close_rc;
}

/// @todo Deprecated.
int lbann::openwrite(const char *file) {
  // define mode (permissions) for new file
  mode_t mode_file = S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP;
  // open the file for writing
  int fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, mode_file);
  if (fd == -1) {
    fprintf(stderr, "ERROR: Failed to create file `%s' (%d: %s) @ %s:%d\n",
            file, errno, strerror(errno), __FILE__, __LINE__
           );
    fflush(stderr);
  }
  return fd;
}

/// @todo Deprecated.
int lbann::closewrite(int fd, const char *file) {
  // fsync file
  int fsync_rc = fsync(fd);
  if (fsync_rc == -1) {
    fprintf(stderr, "ERROR: Failed to fsync file `%s' (%d: %s) @ %s:%d\n",
            file, errno, strerror(errno), __FILE__, __LINE__
           );
    fflush(stderr);
  }

  // close file
  int close_rc = close(fd);
  if (close_rc == -1) {
    fprintf(stderr, "ERROR: Failed to close file `%s' (%d: %s) @ %s:%d\n",
            file, errno, strerror(errno), __FILE__, __LINE__
           );
    fflush(stderr);
  }

  return close_rc;
}
