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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_store/data_store_merge_samples.hpp"
#include "lbann/data_readers/data_reader_merge_samples.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"

#include <sys/stat.h>

namespace lbann {

data_store_merge_samples::~data_store_merge_samples() {
  MPI_Win_free( &m_win );
}

void data_store_merge_samples::setup(bool test_dynamic_cast, bool run_tests) {
  if (m_rank == 0) std::cerr << "STARTING data_store_merge_samples::setup()\n"; 
  //double tm1 = get_time();

  generic_data_store::setup();

/*
  bool run_tests = false;
  if (options::get()->has_bool("test_data_store") && options::get()->get_bool("test_data_store")) {
    run_tests = true;
  }
  */

  if (m_rank == 0) {
    std::cout << "starting data_store_merge_samples::setup() for data reader with role: " << m_reader->get_role() << std::endl;
  }
  
  if (! m_in_memory) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
    throw lbann_exception(err.str());
    //m_buffers.resize( omp_get_max_threads() );
  } 
  
  else {
    //sanity check
    data_reader_merge_samples *reader = dynamic_cast<data_reader_merge_samples*>(m_reader);
    if (reader == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<merge_samples_reader*>(m_reader) failed";
      throw lbann_exception(err.str());
    }
  }
}


}  // namespace lbann
