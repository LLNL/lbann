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

#include "lbann/proto/lbann_proto.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

using google::protobuf::io::FileOutputStream;

using namespace std;
using namespace lbann;


lbann_proto * lbann_proto::s_instance = new lbann_proto;

lbann_proto::lbann_proto() {
}

lbann_proto::~lbann_proto() {
  delete s_instance;
}

void lbann_proto::writePrototextFile(const char *fn) {
  int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  if (not google::protobuf::TextFormat::Print(m_pb, output)) {
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " lbann_proto::writePrototextFile() - failed to write prototext file: " << fn;
    throw runtime_error(err.str());
  }
  delete output;
  close(fd);

/*
  string s;
  google::protobuf::TextFormat::PrintToString(m_pb, &s);
  cout << "as string: " << s << endl;
  */
}


