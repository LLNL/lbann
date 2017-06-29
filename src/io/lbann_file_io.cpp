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

#include "lbann/io/file_io.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "mpi.h"

static mode_t mode_dir = S_IRWXU | S_IRWXG;
static MPI_Comm comm = MPI_COMM_WORLD;

/* creates directory given in dir (absolute path),
 * rank 0 creates directory, all other procs get result via bcast,
 * returns 1 if dir was created, 0 otherwise */
int lbann::makedir(const char *dir) {
  // get our rank
  int rank;
  MPI_Comm_rank(comm, &rank);

  // have rank 0 create directory
  int mkdir_rc;
  if (rank == 0) {
    mkdir_rc = mkdir(dir, mode_dir);
    if (mkdir_rc != 0) {
      if (errno == EEXIST) {
        // not an error if the directory already exists
        mkdir_rc = 0;
      } else {
        fprintf(stderr, "ERROR: Failed to create directory `%s' (%d: %s) @ %s:%d\n",
                dir, errno, strerror(errno), __FILE__, __LINE__
               );
        fflush(stderr);
      }
    }
  }

  // bcast whether directory was created or not
  MPI_Bcast(&mkdir_rc, 1, MPI_INT, 0, comm);

  // return 1 if dir was created successfully
  int ret = (mkdir_rc == 0);
  return ret;
}

int lbann::exists(const char *file) {
  // get our rank
  int rank;
  MPI_Comm_rank(comm, &rank);

  // check whether file exists
  struct stat buffer;
  int exists = 0;
  if (rank == 0) {
    // TODO: would be nice to use something lighter weight than stat here
    if (stat(file, &buffer) == 0) {
      exists = 1;
    }
  }
  MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return exists;
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
