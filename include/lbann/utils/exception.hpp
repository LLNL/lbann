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
//
// lbann_exception .hpp .cpp - LBANN exception class
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_EXCEPTION_HPP_INCLUDED
#define LBANN_EXCEPTION_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/stack_trace.hpp"
#include <iostream>
#include <exception>

// Macro to throw an LBANN exception
#define LBANN_ERROR(message)                                    \
  do {                                                          \
    std::stringstream ss_LBANN_ERROR;                           \
    ss_LBANN_ERROR << "LBANN error "                            \
                   << "(" << __FILE__ << ":" << __LINE__ << ")" \
                   << ": " << (message);                        \
    throw lbann::lbann_exception(ss_LBANN_ERROR.str());         \
  } while (0)

// Macro to print a warning to standard error stream.
#define LBANN_WARNING(message)                                          \
  do {                                                                  \
    int rank_LBANN_WARNING = 0;                                         \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_LBANN_WARNING);                 \
    std::stringstream ss_LBANN_WARNING;                                 \
    ss_LBANN_WARNING << "LBANN warning "                                \
                     << "on rank " << rank_LBANN_WARNING << " "         \
                     << "(" << __FILE__ << ":" << __LINE__ << ")"       \
                     << ": " << (message) << std::endl;                 \
    std::cerr << ss_LBANN_WARNING.str();                                \
  } while (0)

namespace lbann {

class lbann_exception : public std::exception {
 public:
  lbann_exception(const std::string m="my custom exception"):msg(m) { 
    stack_trace::print_lbann_exception_stack_trace(msg);
  }

  ~lbann_exception() override {}
  const char *what() const noexcept override {
    return msg.c_str();
  }

 private:
  std::string msg;
};

inline void lbann_report_exception( lbann_exception& e, lbann_comm *comm=nullptr, std::ostream& os=std::cerr) {
  if( std::string(e.what()) != "" ) {
    if(comm != nullptr) {
      os << "LBANN: rank " << comm->get_rank_in_model() << " of model " << comm->get_model_rank() <<" caught error message:";
    } else {
      os << "LBANN: caught error message:";
    }
    os << "\t" << e.what() << std::endl;
  }
  El::mpi::Abort( El::mpi::COMM_WORLD, 1 );
}
}


#endif // LBANN_EXCEPTION_HPP_INCLUDED
