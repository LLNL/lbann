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
#ifndef __STACK_TRACE_HPP__
#define __STACK_TRACE_HPP__

#include <csignal>
#include <string>

namespace lbann {
namespace stack_trace {

/**
 * Default behaviour: when a signal is caught or an lbann exception
 * thrown, processes write to cerr. The cmd line option
 * "--stack_trace_to_file" will additionally cause each processor
 * to attempt to write its stack trace to
 *    "stack_trace_<rank_in_world>.txt"
 */

/** For internal use; made public for testing.
 *  called by initialize()
 */
void set_lbann_stack_trace_world_rank(int rank);

/** For internal use; made public for testing.
 *  Attempts to print a stack trace. Calls non-reentrant
 *  functions, so behaviour may be undefined if used within
 *  a signal handler. Can be used when exceptions are thrown
 *  without undefined behavior.
 */
void print_stack_trace();


/** For internal use; made public for testing.
 *  Attempts to print a stack trace when an lbann_exception
 *  is thrown. Called by lbann_exception() ctor.
 */
void print_lbann_exception_stack_trace(std::string m);

/** For internal use; made public for testing.
 * called by print_lbann_signal_stack_trace().
 */ 
std::string sig_name(int signal);


/** Register our signal handler; This is called in initialize(),
 *  but only if the cmd line option "--catch_signals" is present.
 *  It is not active by default, because the handler calls non-reentrant,
 *  functions, which can result in undefined behavior. 
 *  See: https://www.ibm.com/developerworks/library/l-reent/
 *  for discussion. That said, it is possible (likely?) that our
 *  handler will work correctly. And if there's a SIGSEV and it
 *  doesn't, nothing much is lost (IMO). 
 */
void register_handler();

/** For internal use; made public for testing. */
void lbann_signal_handler(int signal);


} //namespace stack_trace 
} //namespace lbann


#endif
