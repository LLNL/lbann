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

/** Get human-readable description of signal.
 *  See /usr/include/bits/signum.h for signal explanations.
 */
std::string signal_description(int signal) {

  // Get signal description
  std::string desc;
  switch (signal) {
  case 1:  desc = "hangup";                     break;
  case 2:  desc = "interrupt";                  break;
  case 3:  desc = "quit";                       break;
  case 4:  desc = "illegal instruction";        break;
  case 5:  desc = "trace trap";                 break;
  case 6:  desc = "abort";                      break;
  case 7:  desc = "BUS error";                  break;
  case 8:  desc = "floating-point exception";   break;
  case 9:  desc = "kill, unblockable";          break;
  case 10: desc = "user-defined signal 1";      break;
  case 11: desc = "segmentation violation";     break;
  case 12: desc = "user-defined signal 2";      break;
  case 13: desc = "broken pipe";                break;
  case 14: desc = "alarm clock";                break;
  case 15: desc = "termination";                break;
  case 16: desc = "stack fault";                break;
  case 17: desc = "child status has changed";   break;
  case 18: desc = "continue";                   break;
  case 19: desc = "stop, unblockable";          break;
  case 20: desc = "keyboard stop";              break;
  case 21: desc = "background read from tty";   break;
  case 22: desc = "background write to tty";    break;
  case 23: desc = "urgent condition on socket"; break;
  case 24: desc = "CPU limit exceeded";         break;
  case 25: desc = "file size limit exceeded";   break;
  case 26: desc = "virtual alarm clock";        break;
  case 27: desc = "profiling alarm clock";      break;
  case 28: desc = "window size change";         break;
  case 29: desc = "pollable event occured";     break;
  case 30: desc = "power failure restart";      break;
  case 31: desc = "bad system call";            break;
  }

  // Construct signal description
  std::stringstream ss;
  ss << "signal " << signal;
  if (!desc.empty()) { ss << " (" << desc << ")"; }
  return ss.str();
  
}

/** Whether to write to file when a signal is detected. */
bool write_to_file_on_signal = false;

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
  if (write_to_file_on_signal) {
    ss.clear();
    ss.str("stack_trace");
    if (rank >= 0) { ss << "_rank" << rank; }
    ss << ".txt";
    std::ofstream fs(ss.str().c_str());
    e.print_report(fs);
  }

  // Terminate program
  El::mpi::Abort(El::mpi::COMM_WORLD, 1);
  
}

} // namespace

void register_signal_handler(bool write_to_file) {
  write_to_file_on_signal = write_to_file;
  static struct sigaction sa;
  sa.sa_handler = &handle_signal;
  sa.sa_flags = SA_RESTART;
  sigfillset(&sa.sa_mask);
  const int num_signals = 40;
  for (int i = 0; i < num_signals; i++) {
    sigaction(i, &sa, nullptr);
  }
}

} //namespace stack_trace 
} //namespace lbann
