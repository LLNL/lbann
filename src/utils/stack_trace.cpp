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

#include "lbann/utils/stack_trace.hpp"
#include "lbann/utils/options.hpp"
#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>


namespace lbann {

namespace stack_trace {

static std::ofstream to_file;

static int my_lbann_tracing_id = 0;

struct sigaction sa;

void set_lbann_stack_trace_world_rank(int rank) {
  my_lbann_tracing_id = rank;
}

// optionally opens "to_file" for writing; returns 'false' if
// we tried to open the file, but failed
static bool open_output_file() {
  options * opts = options::get();
  bool success = true;
  if (opts->has_bool("stack_trace_to_file") && opts->get_bool("stack_trace_to_file")) {
    std::stringstream b;
    b << "stack_trace_" << my_lbann_tracing_id << ".txt";
    to_file.open(b.str().c_str());
    if (! to_file.is_open()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " failed to open file: " << b.str() << " for writing";

      //todo: can this be done better? Can't throw exception, else
      //      we go into an infinite loop
      std::cerr << err.str() << std::endl;
      success = false;
    }
  }
  return success;
}

static void close_output_file() {
  if (to_file.is_open()) {
    to_file.close();
  }
}

void print_stack_trace() {
  if (to_file.is_open()) {
    to_file << "\n**************************************************************************\n";
  }

  #define MAX_STACK_FRAMES 64
  static void *stack_traces[MAX_STACK_FRAMES];
  int trace_size = backtrace(stack_traces, MAX_STACK_FRAMES);
  char **messages = backtrace_symbols(stack_traces, trace_size);
  Dl_info info;
  for (int i=0; i<trace_size; i++) {
    std::cerr << "rank: " << my_lbann_tracing_id << " :: ";
    if (to_file.is_open()) {
      to_file << "  >>>> ";
    }
    dladdr(stack_traces[i], &info);
    if (info.dli_sname != NULL) {
      char *demangled_name = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, nullptr);
      if (demangled_name != NULL) {
        std::cerr << demangled_name << std::endl;
        if (to_file.is_open()) {
          to_file << demangled_name << std::endl;
        }
        free(demangled_name);
      } else {
        std::cerr << "demangling failed for: " << info.dli_sname << std::endl;
        if (to_file.is_open()) {
          to_file << "demangling failed for: " << info.dli_sname << std::endl;
        }
      }
    } else {
      std::cerr << "dli_sname == NULL for: " << stack_traces[i] << " backtrace message was: "
                << "     " << messages[i] << std::endl;
      if (to_file.is_open()) {
        to_file << "dli_sname == NULL for: " << stack_traces[i] << " backtrace message was: "
            << "     " << messages[i] << std::endl;
      }
    }
  }
  std::cerr << std::endl;

  if (to_file.is_open()) {
    to_file.close();
  }

  std::cerr << "sleeping for two seconds to give all procs a chance to write ...\n";
  sleep(2);
}


void print_lbann_exception_stack_trace(std::string m) {
  if (! open_output_file()) {
    return;
  }
  
  std::stringstream s;
  s << "\n**************************************************************************\n"
    << " This lbann_exception is about to be thrown:" << m << "\n\n"
    << " Am now attemptting to print the stack trace ...\n"
    << "**************************************************************************\n";
  if (to_file.is_open()) {
    to_file << s.str();
  }
  std::cerr << s.str();
  print_stack_trace();
  close_output_file();
}


std::string sig_name(int signal) {
  std::string r;
  switch (signal) {
    case 1 : 
      r = "SIGHUP"; break;
    case 2 :
      r = "SIGINT"; break;
    case 3 :
      r = "SIGQUIT"; break;
    case 4 :
      r = "SIGILL"; break;
    case 5 :
      r = "SIGTRAP"; break;
    case 6 :
      r = "SIGABRT"; break;
    case 7 :
      r = "SIGBUS"; break;
    case 8 :
      r = "SIGFPE"; break;
    case 9 :
      r = "SIGKILL"; break;
    case 10 :
      r = "SIGUSR1"; break;
    case 11 :
      r = "SIGSEGV"; break;
    case 12 :
      r = "SIGUSR2"; break;
    case 13 :
      r = "SIGPIPE"; break;
    case 14 :
      r = "SIGALRM"; break;
    case 15 :
      r = "SIGTERM"; break;
    case 16 :
      r = "SIGSTKFLT"; break;
    case 17 :
      r = "SIGCHLD"; break;
    case 18 :
      r = "SIGCONT"; break;
    case 19 :
      r = "SIGSTOP"; break;
    case 20 :
      r = "SIGTSTP"; break;
    case 21 :
      r = "SIGTTIN"; break;
    case 22 :
      r = "SIGTTOU"; break;
    case 23 :
      r = "SIGURG"; break;
    case 24 :
      r = "SIGXCPU"; break;
    case 25 :
      r = "SIGXFSZ"; break;
    case 26 :
      r = "SIGVTALRM"; break;
    case 27 :
      r = "SIGPROF"; break;
    case 28 :
      r = "SIGWINCH"; break;
    case 29 :
      r = "SIGPOLL"; break;
    case 30 :
      r = "SIGPWR"; break;
    case 31 :
      r = "SIGSYS"; break;
    default :
      std::stringstream s;
      s << "unknown signal #" << signal;
      r = s.str();
  }
  return r;
}

void register_handler() {
  options *opts = options::get();
  if (opts->has_bool("catch_signals") and opts->get_bool("catch_signals")) {
    sa.sa_handler = &lbann_signal_handler;
    sa.sa_flags = SA_RESTART;
    sigfillset(&sa.sa_mask);

    for (int i=0; i<40; i++) {
      sigaction(i, &sa, NULL);
    }
  }
}


void lbann_signal_handler(int signal) {
std::cerr << "starting lbann_signal_handler\n";
  if (! open_output_file()) {
    return;
  }
  std::stringstream s;
  s <<  
         "\n**************************************************************************\n"
         " Caught this signal: " << sig_name(signal) << "\n"
         " Note: see /usr/include/bits/signum.h for signal explanations\n"
         " Am now attemptting to print the stack trace ...\n"
         "**************************************************************************\n";
  if (to_file.is_open()) {
    to_file << s.str();
  }
  std::cerr << s.str();
  print_stack_trace();
  close_output_file();
}

} //namespace stack_trace 
} //namespace lbann
