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

#include "lbann/lbann.hpp"
#include <csignal>


using namespace lbann;

const int lbann_default_random_seed = 42;

class A {
  public :
    void testme_class_A_one() {
      std::cerr << "starting testme_class_A_one()\n";
      std::stringstream s;
      s << __FILE__ << " " << __LINE__ << " :: "
        << " A:testme_class_A_one() is throwing an exception for testing";
      throw lbann_exception(s.str());
    }

    void testme_class_A_two() {
      std::cerr << "starting testme_class_A_two()\n";
      testme_class_A_one();
    }
};

class B {
  public :
    void testme_class_B_the_first() {
      std::cerr << "starting testme_class_B_the_first()\n";
      A a;
      a.testme_class_A_two();
    }

    void testme_class_B_sigsegv() {
      raise(SIGSEGV);
    }

    void testme_class_B_sigint() {
      raise(SIGINT);
    }
};

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    //must be called after opts->init(); must also specify "--catch-signals"
    //on cmd line
    stack_trace::register_handler();

    if (master) {
      std::cerr << "running test for stack tracing when sigsegv is raised\n\n";
    }

    B b;
    b.testme_class_B_sigsegv();

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  }

  finalize(comm);
  return 0;
}
