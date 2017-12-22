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

#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "lbann/utils/options.hpp"
#include <unistd.h>

namespace lbann {

static int world_rank = 0;

void set_lbann_exception_world_rank(int rank) {
  world_rank = rank;
}

void print_lbann_exception_stack_trace(std::string m) {

  std::cerr << 
    "\n**************************************************************************\n"
      "* an lbann_exception(...) was thrown; the stack trace follows below.     *\n"
      "* note: the cmd line option \"--exception_to_file\" will cause each      *\n"
      "*       process to write it's call stack to file:                        *\n"
      "*                   exception_call_stack_<rank>.txt                      *\n"
      "**************************************************************************\n";

  std::cerr << "The exception (which will be caught by main) is:\n"
            << m << std::endl << std::endl;

  std::ofstream out;
  options *opts = options::get();
  if (opts->has_bool("exception_to_file") && opts->get_bool("exception_to_file")) {
    std::stringstream s;
    s << "exception_call_stack_" << world_rank << ".txt";
    out.open(s.str().c_str());
    out << "The following exception is being thrown:\n"
        << "    " << m << "\n\nStack trace follows:\n";
  }

  #define MAX_STACK_FRAMES 64
  static void *stack_traces[MAX_STACK_FRAMES];
  int trace_size = backtrace(stack_traces, MAX_STACK_FRAMES);
  char **messages = backtrace_symbols(stack_traces, trace_size);
  Dl_info info;
  for (int i=0; i<trace_size; i++) {
    std::cerr << "rank: " << world_rank << " :: ";
    if (out.is_open()) {
      out << "  >>>> ";
    }
    dladdr(stack_traces[i], &info);
    if (info.dli_sname != NULL) {
      char *demangled_name = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, nullptr);
      if (demangled_name != NULL) {
        std::cerr << demangled_name << std::endl;
        if (out.is_open()) {
          out << demangled_name << std::endl;
        }
        free(demangled_name);
      } else {
        std::cerr << "demangling failed for: " << info.dli_sname << std::endl;
        if (out.is_open()) {
          out << "demangling failed for: " << info.dli_sname << std::endl;
        }
      }
    } else {
      std::cerr << "dli_sname == NULL for: " << stack_traces[i] << " backtrace message was: "
                << "     " << messages[i] << std::endl;
      if (out.is_open()) {
        out << "dli_sname == NULL for: " << stack_traces[i] << " backtrace message was: "
            << "     " << messages[i] << std::endl;
      }
    }
  }
  std::cerr << std::endl;

  if (out.is_open()) {
    out.close();
  }

  std::cerr << "sleeping for two seconds to give all procs a chance to write ...\n";
  sleep(2);
}


} //namespace lbann
