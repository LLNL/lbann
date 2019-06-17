////////////////////////////////////////////////////////////////////////////////xecu
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
////////////////////////////////////////////////////////////////////////////////

#ifndef __CYG_PROFILE_HPP__
#define __CYG_PROFILE_HPP__

#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <cxxabi.h>

/**
 * This module contains pure C variables and functions that are used
 * in conjunction with __cyg_profile_func_enter() and __cyg_profile_func_exit()
 * to profile the stack at runtime. Some functions are inlined below; others
 * are defined in src/utils/stack_profiler.cpp
 */

extern "C" {

    extern FILE *c_hash_fp_full_stack_trace;
    extern FILE *c_hash_fp_full_stack_trace_metadata;
    extern int c_hash_thread_id;
    extern int c_hash_profiling_is_turned_on;
    extern char **c_hash_default_discard_strings;
    extern int c_hash_num_default_discard_items;

    extern short c_hash_func_id;

    extern short c_hash_depth;
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
        short id;
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


    void __cyg_profile_func_enter (void *func, void *caller) __attribute__((no_instrument_function));
    void __cyg_profile_func_exit (void *func, void *caller) __attribute__((no_instrument_function));

#define HASH_1(k,size,idxOut)                   \
    {  *idxOut = k % size;  }

#define HASH_2(k,size,idxOut)                   \
    {                                           \
        long r = k % (size-13);                 \
        r = (r % 2) ? r : r+1;                  \
        *idxOut = r;                            \
    }

    //! inserts "func" in a hash table slot. WARNING:
    //! this is for internal use; end users should instead
    //! call "c_hash_insert_or_update"
    static void c_hash_insert(void *func) __attribute__((no_instrument_function));

    //! returns a pointer to the slot that has the (hashed) key, "name,"
    //! or NULL if it doesn't exist
    static c_hash_node * c_hash_lookup(void *func) __attribute__((no_instrument_function));

    //! create a slot in the table with the data: func, count=1;
    //! if a slot with the name already exists, increment the count
    static void c_hash_insert_or_update(void *func)__attribute__((no_instrument_function));

    //! well known hash function (google it) Hashes a string to an integer
    static unsigned long long djb2_hash(void *, int table_size)  __attribute__((no_instrument_function));


    inline void __cyg_profile_func_exit (void *func, void *caller)
    {
        if (c_hash_thread_id != 0) {
            return;
        }
        if (c_hash_profiling_is_turned_on == 0) {
            return;
        }
        --c_hash_depth;
    }


    //! well known hash function (google it) Hashes a string to an integer
    static unsigned long long djb2_hash(void *func, int table_size)
    {
        char *str = (char*)&func;
        unsigned long long hash = 5381;
        int d = sizeof(func);
        for (int j=0; j<d; j++) {
            hash = ((hash << 5) + hash) + (int)str[j]; /* hash * 33 + c */
        }
        return hash;
    }

    inline void __cyg_profile_func_enter (void *func, void *caller)
    {
        if (c_hash_profiling_is_turned_on == 0) {
            return;
        }
        if (c_hash_thread_id != 0) {
            return;
        }
        ++c_hash_depth;
        c_hash_insert_or_update(func);
    }


    //! insert "func" in a hash table slot. WARNING:
    //! this is for internal use; end users should instead
    //! call "c_hash_insert_or_update"
    static void c_hash_insert(void *func)
    {
        c_hash_table *h = c_hash_the_hash_table;
        h->count += 1;
        if (1000 + h->count == h->size) {
            printf("out of slots! Please reimplement rehash\n");
            exit(9);
        }

        dladdr(func, &c_hash_info);
        if (c_hash_info.dli_sname != NULL) {
            int success = 0;
            char *demangled_name = abi::__cxa_demangle(c_hash_info.dli_sname, nullptr, nullptr, nullptr);
            if (demangled_name != NULL) {
                unsigned long long key = djb2_hash(func, h->size);
                unsigned long start, tmp;
                HASH_1(key, h->size, &start);
                for (long i=0; i<h->size; i++) {
                    HASH_2(key, h->size, &tmp);
                    long idx = (start + i*tmp) % h->size;
                    if (h->data[idx].mark < h->cur_mark) {
                        h->data[idx].key = key;
                        h->data[idx].mark = h->cur_mark;
                        h->data[idx].count = 1;
                        h->data[idx].name = (char*)malloc(strlen(demangled_name)+1);
                        strcpy(h->data[idx].name, demangled_name);
                        h->data[idx].func = func;
                        h->data[idx].id = c_hash_func_id;
                        if (c_hash_fp_full_stack_trace_metadata != NULL) {
                            fprintf(c_hash_fp_full_stack_trace_metadata, "%hd %s\n", c_hash_func_id, demangled_name);
                            fwrite(&c_hash_func_id, sizeof(short), 1, c_hash_fp_full_stack_trace);
                            fwrite(&c_hash_depth, sizeof(short), 1, c_hash_fp_full_stack_trace);
                            ++c_hash_func_id;
                        }
                        success = 1;
                        break;
                    }
                }
                if (success != 1) {
                    printf("\n\nERROR: success = %d\n\n", success);
                    exit(9);
                }
            }
            free(demangled_name);
        }
    }

    //! returns a pointer to the slot that has the (hashed) key, "name,"
    //! or NULL if it doesn't exist
    c_hash_node * c_hash_lookup(void *func)
    {
        ++c_hash_lookups;
        c_hash_node * retval = NULL;
        c_hash_table *h = c_hash_the_hash_table;
        unsigned long long key = djb2_hash(func, h->size);

        unsigned long start, tmp;
        HASH_1(key, h->size, &start);
        for (long i=0; i<h->size; i++) {
            if (i > 0) ++c_hash_collisions;
            HASH_2(key, h->size, &tmp);
            long idx = (start + i*tmp) % h->size;
            if (h->data[idx].mark != h->cur_mark) {
                break;
            } else {
                if (func == h->data[idx].func) {
                    retval = &h->data[idx];
                }
            }
        }
        return retval;
    }

    //! create a slot in the table with the data: name, count=1, time=0.0;
    //! if a slot with the name already exists, increment the count, and
    //! time += time
    static void c_hash_insert_or_update(void *func)
    {
        c_hash_node *c;
        if ( (c = c_hash_lookup(func)) != NULL) {
            c->count += 1;
            if (c_hash_fp_full_stack_trace_metadata != NULL) {
                fwrite(&(c->id), sizeof(short), 1, c_hash_fp_full_stack_trace);
                fwrite(&c_hash_depth, sizeof(short), 1, c_hash_fp_full_stack_trace);
            }
        } else {
            c_hash_insert(func);
        }
    }

} // extern "C"

#endif // LBANN_BASE_HPP
