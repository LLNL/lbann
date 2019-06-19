#include "lbann/utils/options.hpp"
#include "lbann/utils/stack_profiler.hpp"
#include "lbann/utils/cyg_profile.hpp"
#include <cxxabi.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string.h>

/**
 * following are pure C functions for creating a hash table,
 * and definitions for variables used in the 'extern "C"' inline
 * functions in lbann/base.hpp
 *
 * following that are the stack_profiler class definitions;
 * the stack_profiler is a c++ class in the lbann namespace
 */

c_hash_table * c_hash_the_hash_table = NULL;
FILE *c_hash_fp_full_stack_trace = NULL;
FILE *c_hash_fp_full_stack_trace_metadata = NULL;
int c_hash_profiling_is_turned_on = 0;
int c_hash_thread_id;
short c_hash_func_id = 0;

Dl_info c_hash_info;
short c_hash_depth = 0;

long c_hash_lookups = 0;
long c_hash_collisions = 0;

void c_hash_set_thread_id(int id) {
  c_hash_thread_id = id;
}

void c_hash_create(long size) {
  c_hash_table *h = (struct c_hash_table*) malloc(sizeof(struct c_hash_table));
  h->size = 0;
  h->count = 0;
  h->cur_mark = 1;
  h->data = NULL;

  //initial size for hash table; want this to be a power of 2
  long sz = 16;
  while (sz < size) { sz *= 2; }
  // rule-of-thumb: ensure there's some padding
  if ( (sz-size) < (.1 * sz) ) { sz *= 2.0; }
  h->size = sz;

  //allocate and zero the hash table
  c_hash_node *d = h->data = (c_hash_node*)malloc(sz*sizeof(c_hash_node));
  for (int i=0; i<sz; i++) {
    d[i].key = 0;
    d[i].mark = 0;
    d[i].func = 0;
  }

  c_hash_the_hash_table = h;
}

//===========================================================================

namespace lbann {

stack_profiler * stack_profiler::s_instance = new stack_profiler;

stack_profiler::~stack_profiler() {
  //todo: free hash table memory
  if (c_hash_fp_full_stack_trace != NULL) {
    fclose(c_hash_fp_full_stack_trace);
  }
  if (c_hash_fp_full_stack_trace_metadata != NULL) {
    fclose(c_hash_fp_full_stack_trace_metadata);
  }
}

stack_profiler::stack_profiler()
  : m_full_stack_trace(false)
  {}

void stack_profiler::activate(int thread) {
  m_thread_id = thread;
  c_hash_thread_id = thread;
  options *opts = options::get();

  if (opts->get_bool("st_on")) {
    std::cerr << "creating hash table!\n";
    c_hash_create(10000);
    c_hash_profiling_is_turned_on = 1;
    if (opts->get_bool("st_full_trace")) {
      m_full_stack_trace = true;
      if (m_thread_id == 0) {
        c_hash_fp_full_stack_trace = fopen("full_stack_trace.bin", "wb");
        c_hash_fp_full_stack_trace_metadata = fopen("full_stack_trace.txt", "w");
      }
    }
  } else {
    c_hash_profiling_is_turned_on = 0;
  }
}

bool count_sorter (const std::pair<std::string, long> &a, const std::pair<std::string, long> &b) {
  return a.second > b.second;
}

void stack_profiler::print() {
  if (m_thread_id != 0) { return; }
  c_hash_table *h = c_hash_the_hash_table;
  if (h == NULL) {
    return;
  }
  std::cout << std::endl << "demangling ... num hash table entries: " << h->count << "\n";

  std::vector<std::pair<std::string, long>> v;
  v.reserve( h->count );
  c_hash_node *data = h->data;
  for (long k=0; k < h->size; k++) {
    if (data[k].mark == h->cur_mark) {
      v.push_back(std::make_pair(data[k].name, data[k].count));
    }
  }

  std::sort(v.begin(), v.end(), count_sorter);
  for (auto t : v) {
    std::cout << t.second << "  " << t.first << std::endl;
  }

  std::cout << "===================================================================\n";
  std::cout << "hash table lookups:    " << c_hash_lookups << std::endl;
  std::cout << "hash table collisions: " << c_hash_collisions << std::endl;
}

} // namespace lbann


