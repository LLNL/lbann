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

#include "lbann/io/file_io.hpp"
#include "lbann/utils/exception.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstring>
#include <cstdio>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <sstream>

#include "mpi.h"

/* creates directory given in dir (absolute path),
 * returns 1 if dir was created, 0 otherwise */
int lbann::makedir(const char *dir) {
  int mkdir_rc = mkdir(dir, S_IRWXU | S_IRWXG);
  if (mkdir_rc == 0 || errno == EEXIST) {
    return 1;
  } else {
    std::stringstream err;
    err << "failed to create directory (" << dir << ") "
        << "with error " << errno << " (" << strerror(errno) << ")";
    LBANN_ERROR(err.str());
    return 0;
  }
}

int lbann::exists(const char *file) {
  struct stat buffer;
  return (stat(file, &buffer) == 0) ? 1 : 0;
}

int lbann::openread(const char *file) {
  // open the file for writing
  int fd = open(file, O_RDONLY);
  if (fd == -1) {
  }
  return fd;
}

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
