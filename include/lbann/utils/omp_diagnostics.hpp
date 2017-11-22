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
#ifndef LBANN_OMP_DIAGNOSTICS_HPP
#define LBANN_OMP_DIAGNOSTICS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>           // sysconf
#include <stdint.h>
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

#ifdef MPI_INCLUDED
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

namespace lbann {
int get_num_pus();
int get_affinity(uint8_t *cpus, uint8_t *count);
void th_print_affinity(int rank, int np, char *host);
void print_affinity(int rank, int np, char *host);
int get_env_var(const char *id);
int get_sleep_sec();
void print_affinity_subset(int rank, int np, char *host);
void display_omp_setup();
} // namespace lbann
#endif // LBANN_OMP_DIAGNOSTICS_HPP
