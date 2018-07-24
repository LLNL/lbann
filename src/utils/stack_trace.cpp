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

  // Stack frames to ignore
  const std::vector<std::string> ignored_frames
    = {"lbann::stack_trace",
       "lbann::lbann_exception"};
  
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
      } else if (std::all_of(ignored_frames.begin(),
                             ignored_frames.end(),
                             [name](const std::string& str) {
                               return str.find(name) == std::string::npos;
                             })) {
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

/** Get human-readable name for signal.
 *  See /usr/include/bits/signum.h for signal explanations.
 */
std::string signal_name(int signal) {
  switch (signal) {
  case 1:  return "SIGHUP";
  case 2:  return "SIGINT";
  case 3:  return "SIGQUIT";
  case 4:  return "SIGILL";
  case 5:  return "SIGTRAP";
  case 6:  return "SIGABRT";
  case 7:  return "SIGBUS";
  case 8:  return "SIGFPE";
  case 9:  return "SIGKILL";
  case 10: return "SIGUSR1";
  case 11: return "SIGSEGV";
  case 12: return "SIGUSR2";
  case 13: return "SIGPIPE";
  case 14: return "SIGALRM";
  case 15: return "SIGTERM";
  case 16: return "SIGSTKFLT";
  case 17: return "SIGCHLD";
  case 18: return "SIGCONT";
  case 19: return "SIGSTOP";
  case 20: return "SIGTSTP";
  case 21: return "SIGTTIN";
  case 22: return "SIGTTOU";
  case 23: return "SIGURG";
  case 24: return "SIGXCPU";
  case 25: return "SIGXFSZ";
  case 26: return "SIGVTALRM";
  case 27: return "SIGPROF";
  case 28: return "SIGWINCH";
  case 29: return "SIGPOLL";
  case 30: return "SIGPWR";
  case 31: return "SIGSYS";
  default: return "signal " + std::to_string(signal);
  }
}

/** Whether to write to file when a signal is detected. */
bool write_to_file_on_signal = false;
  
/** Signal handler.
 *  Output signal name and stack trace to standard output and to a
 *  file (if desired).
 */
void handle_signal(int signal) {
  std::stringstream ss;
  ss << "Caught " << signal_name(signal);
  const auto& rank = get_rank_in_world();
  if (rank >= 0) { ss << " on rank " << rank; }
  throw exception(ss.str());
}

} // namespace

void register_signal_handler(bool write_to_file) {
  write_to_file_on_signal = write_to_file;
  static struct sigaction sa;
  sa.sa_handler = &handle_signal;
  sa.sa_flags = SA_RESTART;
  sigfillset(&sa.sa_mask);
  for (int i=0; i<40; i++) {
    sigaction(i, &sa, NULL);
  }
}

} //namespace stack_trace 
} //namespace lbann
