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

#include "lbann/data_store/data_store_imagenet.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void data_store_imagenet::setup() {
  generic_data_store::setup();
  if (! m_in_memory) {
    m_buffers.resize( omp_get_max_threads() );
  }
}

void data_store_imagenet::get_data_buf(std::string dir, std::string filename, std::vector<unsigned char> *&buf, int tid) {
  static int idx = 0;
  std::vector<unsigned char> &b = m_buffers[tid];
  std::string imagepath = dir + filename;
  if (m_master && idx == 0) {
    std::cerr << "\ndata_store_imagenet: READING: " << imagepath << std::endl << std::endl;
    ++idx;
  }  
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (! in.good()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << imagepath << " for reading";
    throw lbann_exception(err.str());
  }
  in.seekg(0, std::ios::end);
  size_t sz = in.tellg();
  in.seekg(0, std::ios::beg);
  b.resize(sz);
  in.read((char*)&b[0], sz*sizeof(unsigned char));
  in.close();
  buf = &m_buffers[tid];
}

}  // namespace lbann
