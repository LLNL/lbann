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

#include "lbann/utils/stack_trace.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/comm.hpp"
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>
// #include <unistd.h>
#include <csignal>

namespace lbann {
namespace stack_trace {

std::string get() {
  std::stringstream ss;

  // Get stack frames
  std::vector<void*> frames(128, nullptr);
  const auto& frames_size = backtrace(frames.data(), frames.size());
  frames.resize(frames_size, nullptr);

  // Get demangled stack frame names
  auto* symbols = backtrace_symbols(frames.data(), frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    ss << std::setw(4) << i << ": ";
    Dl_info info;
    dladdr(frames[i], &info);
    if (info.dli_sname != nullptr) {
      auto* name = abi::__cxa_demangle(info.dli_sname,
                                       nullptr, nullptr, nullptr);
      if (name == nullptr) {
        ss << info.dli_sname << " (demangling failed)";
      } else {
        ss << name;
      }
      std::free(name);
    } else {
      if (symbols != nullptr) { ss << symbols[i] << " "; }
      ss << "(could not find stack frame symbol)";
    }
    ss << std::endl;
  }
  std::free(symbols);

  return ss.str();
}

namespace {

/** Get human-readable description of signal. */
std::string signal_description(int signal) {

  // Get signal description
  // Note: Multiple signals can share the same code, so we can't use a
  // switch-case statement. Signal descriptions are taken from the
  // POSIX C standard
  // (http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html).
  std::string desc;
#define SIGNAL_CASE(name, description)          \
  do {                                          \
    if (desc.empty() && signal == name) {       \
      desc = #name " - " description;           \
    }                                           \
  } while (false)
  SIGNAL_CASE(SIGABRT, "process abort signal");
  SIGNAL_CASE(SIGALRM, "alarm clock");
  SIGNAL_CASE(SIGBUS,  "access to an undefined portion of a memory object");
  SIGNAL_CASE(SIGCHLD, "child process terminated, stopped");
  SIGNAL_CASE(SIGCONT, "continue executing, if stopped");
  SIGNAL_CASE(SIGFPE,  "erroneous arithmetic operation");
  SIGNAL_CASE(SIGHUP,  "hangup");
  SIGNAL_CASE(SIGILL,  "illegal instruction");
  SIGNAL_CASE(SIGINT,  "terminal interrupt signal");
  SIGNAL_CASE(SIGKILL, "kill (cannot be caught or ignored)");
  SIGNAL_CASE(SIGPIPE, "write on a pipe with no one to read it");
  SIGNAL_CASE(SIGQUIT, "terminal quit signal");
  SIGNAL_CASE(SIGSEGV, "invalid memory reference");
  SIGNAL_CASE(SIGSTOP, "stop executing (cannot be caught or ignored)");
  SIGNAL_CASE(SIGTERM, "termination signal");
  SIGNAL_CASE(SIGTSTP, "terminal stop signal");
  SIGNAL_CASE(SIGTTIN, "background process attempting read");
  SIGNAL_CASE(SIGTTOU, "background process attempting write");
  SIGNAL_CASE(SIGUSR1, "user-defined signal 1");
  SIGNAL_CASE(SIGUSR2, "user-defined signal 2");
  SIGNAL_CASE(SIGTRAP, "trace/breakpoint trap");
  SIGNAL_CASE(SIGURG,  "high bandwidth data is available at a socket");
  SIGNAL_CASE(SIGXCPU, "CPU time limit exceeded");
  SIGNAL_CASE(SIGXFSZ, "file size limit exceeded");
#undef SIGNAL_CASE

  // Construct signal description
  std::stringstream ss;
  ss << "signal " << signal;
  if (!desc.empty()) { ss << " (" << desc << ")"; }
  return ss.str();

}

/** Base name for stack trace output file. */
std::string stack_trace_file_base = "";

/** Signal handler.
 *  Output signal name and stack trace to standard error and to a file
 *  (if desired).
 */
void handle_signal(int signal) {

  // Print error message and stack trace to standard error
  std::stringstream ss;
  ss << "Caught " << signal_description(signal);
  const auto& rank = get_rank_in_world();
  if (rank >= 0) { ss << " on rank " << rank; }
  const exception e(ss.str());
  e.print_report();

  // Print error message and stack trace to file
  if (!stack_trace_file_base.empty()) {
    ss.clear();
    ss.str(stack_trace_file_base);
    if (rank >= 0) { ss << "_rank" << rank; }
    ss << ".txt";
    std::ofstream fs(ss.str().c_str());
    e.print_report(fs);
  }

  // Terminate program
  El::mpi::Abort(El::mpi::COMM_WORLD, 1);

}

} // namespace

void register_signal_handler(std::string file_base) {
  stack_trace_file_base = file_base;

  // Construct signal action object with signal handler
  static struct sigaction sa;
  sa.sa_handler = &handle_signal;
  sa.sa_flags = SA_RESTART;
  sigfillset(&sa.sa_mask);

  // Register signal handler for fatal signals
  std::vector<int> fatal_signals = {SIGABRT, SIGALRM, SIGBUS , SIGFPE ,
                                    SIGHUP , SIGILL , SIGINT , SIGKILL,
                                    SIGPIPE, SIGQUIT, SIGSEGV, SIGTERM,
                                    SIGUSR1, SIGUSR2, SIGTRAP, SIGXCPU,
                                    SIGXFSZ};
  for (const auto& signal : fatal_signals) {
    sigaction(signal, &sa, nullptr);
  }

}

} //namespace stack_trace
} //namespace lbann
