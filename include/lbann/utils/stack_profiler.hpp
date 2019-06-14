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
#ifndef __STACK_PROFILER_HPP__
#define __STACK_PROFILER_HPP__

#include <string.h>
#include <dlfcn.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cxxabi.h>


namespace lbann {

/**
 *  This is a singleton, globally accessible class, for recording
 *  stack traces and timing
 */

class stack_profiler {
public :
  static stack_profiler * get() __attribute__((no_instrument_function)) {
    return s_instance;
  }

  /** Turns on stack profiling *if* you have passed the cmd line
   *  option: --st_on or --st_on=1
   */
  void activate(int thread_id) __attribute__((no_instrument_function));

  void print() __attribute__((no_instrument_function));

private :
  stack_profiler() __attribute__((no_instrument_function));
  ~stack_profiler() __attribute__((no_instrument_function));
  stack_profiler(const stack_profiler &) {}
  stack_profiler& operator=(const stack_profiler&) { return *this; }

  static stack_profiler *s_instance;

  bool m_full_stack_trace;

  int m_thread_id ;

  // This variable is unused, causing a certain clangy compiler to complain
  //Dl_info m_info;

};

} //namespace lbann


#if 0
/**
 * This module contains pure C declaration for stack profiling support
 * for stack tracing
 */

#define IGNORE_BIN_CT 2000000

extern "C" {

extern FILE *c_hash_fp_full_stack_trace;
extern int c_hash_thread_id;
extern int c_hash_profiling_is_turned_on;
extern char **c_hash_default_discard_strings;
extern int c_hash_num_default_discard_items;

extern int c_hash_depth;
extern Dl_info c_hash_info;

extern void **c_hash_discard_void_ptrs;
extern char **c_hash_discard_strings;
extern int *c_hash_discard_counts;
extern int c_hash_discard_void_ptrs_num_active;

extern long c_hash_lookups;
extern long c_hash_collisions;

//! hash table slots
struct c_hash_node {
  unsigned long key;
  long  mark;

  //following are the data fields
  char *name;
  long count;
  void * func;
};

//! hash table structure
struct c_hash_table {
  long size;   //total slots in table
  long count;  //number of insertions in table
  long cur_mark;
  c_hash_node *data;  //typedef struct _hash_node_private HashRecord
};

extern c_hash_table * c_hash_the_hash_table;

//! the hash table! This is a singleton
//! creates the singleton hash table; the table will contain at least
//! "size" slots (FYI, the table size is a power of two)
void c_hash_create(long size) __attribute__((no_instrument_function));

void c_hash_set_thread_id(int id) __attribute__((no_instrument_function));

} // extern "C"

#endif
#endif
