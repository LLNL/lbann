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

/// OpenMP Diagnostic code from Edgar Leon at LLNL

#include "lbann_config.hpp"

#ifdef LBANN_GNU_LINUX

#include "lbann/utils/omp_diagnostics.hpp"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>           // sysconf
#include <omp.h>

/* __USE_GNU is needed for CPU_ISSET definition */
#ifndef __USE_GNU
#define __USE_GNU 1
#endif
#include <sched.h>            // sched_getaffinity

#include "mpi.h"

#ifdef HPM
#include "libhpc.h"
#endif

#ifdef MPI_VERSION
#define MPI_CHECK( arg )			   \
  if ( (arg) != MPI_SUCCESS ) {			   \
    fprintf( stderr, "%s:%d " #arg " failed\n",	   \
	     __FILE__, __LINE__			   \
	     );					   \
  }
#endif

#define NULL_CHECK( arg )					\
  if ( (arg) == NULL ) {					\
    fprintf( stderr, "%s:%d " #arg " NULL return\n",		\
	     __FILE__, __LINE__					\
	     );							\
  }

#define NONZERO_CHECK( arg )				\
  if ( (arg) != 0 ) {					\
    fprintf(stderr, "%s:%d " #arg " NON-ZERO return\n", \
	    __FILE__, __LINE__);			\
  }

#endif // LBANN_GNU_LINUX

namespace lbann {

#ifdef LBANN_GNU_LINUX

/* Get number of processing units (cores or hwthreads) */
int get_num_pus()
{
  int pus;
  if ( (pus = sysconf(_SC_NPROCESSORS_ONLN)) < 0 )
    perror("sysconf");
  return pus;
}

/*
 * Get the affinity.
 */
int get_affinity(uint8_t *cpus, uint8_t *count)
{
  int i, rc;
  cpu_set_t resmask;
  int pus = get_num_pus();


  CPU_ZERO(&resmask);
  if ( (rc = sched_getaffinity(0, sizeof(resmask), &resmask)) < 0 ) {
    perror("sched_getaffinity");
    return rc;
  }

  *count = 0;
  for (i=0; i<pus; i++)
    if ( CPU_ISSET(i, &resmask) ) {
      cpus[*count] = i;
      (*count)++;
    }

  return 0;
}

void th_print_affinity(int rank, int np, char *host)
{
  int nc;
  char c = ',';
  char buf[1024*2];
  uint8_t i, *cpus, count=0;
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();


  cpus = (uint8_t *) malloc(sizeof(uint8_t) * get_num_pus());
  get_affinity(cpus, &count);

  nc = sprintf(buf, "Task %3d/%d Thread %3d/%d running on cpu ",
	       rank, np, tid, nthreads);
  for (i=0; i<count; i++) {
    if (i == count-1)
      c = '\0';
    nc += sprintf(buf+nc, "%d%c", cpus[i], c);
  }
  printf("  %s %s\n", host, buf);
}

void print_affinity(int rank, int np, char *host)
{
  int nc;
  char c = ',';
  char buf[1024*2];
  uint8_t i, *cpus, count=0;


  cpus = (uint8_t *) malloc(sizeof(uint8_t) * get_num_pus());
  get_affinity(cpus, &count);

  nc = sprintf(buf, "Task %3d/%d running on cpus ", rank, np);
  for (i=0; i<count; i++) {
    if (i == count-1)
      c = '\0';
    nc += sprintf(buf+nc, "%d%c", cpus[i], c);
  }
  printf("  %s %s\n", host, buf);
}


int get_env_var(const char *id)
{
  char *buf = getenv(id);

  if (buf == nullptr)
    return 0;
  else
    return atoi(buf);
}

int get_sleep_sec()
{
  char *buf = getenv("SLEEP_SEC");

  if (buf == nullptr)
    return 0;
  else
    return atoi(buf);
}


void print_affinity_subset(int rank, int np, char *host)
{
  MPI_Comm sh_comm;
  int sh_rank, sh_np;


  // Create communicators
  MPI_CHECK( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
				 rank, MPI_INFO_NULL, &sh_comm) );
  MPI_CHECK( MPI_Comm_rank(sh_comm, &sh_rank) );
  MPI_CHECK( MPI_Comm_size(sh_comm, &sh_np) );

  if (rank < 2*sh_np)
    print_affinity(rank, np, host);

  MPI_CHECK( MPI_Comm_free(&sh_comm) );
}
#endif // LBANN_GNU_LINUX

void __attribute__((used)) display_omp_setup()
{
#ifdef LBANN_GNU_LINUX
  int rank, np, secs, len, mpi_only, mpi_subset;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Get_processor_name(hostname, &len);

  secs = get_sleep_sec();
  (void) secs;
  mpi_only = get_env_var("MPI_ONLY");
  mpi_subset = get_env_var("MPI_SUBSET");


  #ifdef HPM
    hpmInit(0, "HPMTest");
    hpmStart(1, "Region 1");
  #endif

  if (rank == 0)
    printf("\n");

  if (mpi_only != 0) {
    if (mpi_subset) {
      print_affinity_subset(rank, np, hostname);
    }else {
      print_affinity(rank, np, hostname);
    }
  }else {
    /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel shared(rank, np, secs, hostname)
    {
      th_print_affinity(rank, np, hostname);
    }  /* All threads join master thread and disband */
  }

#ifdef HPM
  hpmStop(1);
  hpmTerminate(0);
#endif
#endif // LBANN_GNU_LINUX
}
} // namespace lbann
