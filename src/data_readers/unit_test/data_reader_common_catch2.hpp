////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#ifndef __DATA_READER_TEST_COMMON_HPP__
#define __DATA_READER_TEST_COMMON_HPP__

#include <sys/stat.h>  //for mkdir
#include <sys/types.h> //for getpid
#include <sys/types.h> //for mkdir
#include <unistd.h>    //for getpid

#include "lbann/data_readers/data_reader.hpp"
#include <lbann/base.hpp>

#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>
namespace pb = ::google::protobuf;

/** create a directory in /tmp; returns the pathname to the directory */
std::string create_test_directory(std::string base_name);

/** Instantiates one or more data readers from the input 'prototext' string.
 *  Users should ensure that the appropriate options (if any) are set prior
 *  to calling this function, i.e:
 *    lbann::options *opts = lbann::options::get();
 *    opts->set_option("preload_data_store", true);
 */
std::map<lbann::execution_mode, lbann::generic_data_reader*>
instantiate_data_readers(std::string prototext_in,
                         lbann::lbann_comm& comm_in,
                         lbann::generic_data_reader*& train_ptr,
                         lbann::generic_data_reader*& validate_ptr,
                         lbann::generic_data_reader*& test_ptr,
                         lbann::generic_data_reader*& tournament_ptr);

void write_file(std::string data, std::string dir, std::string fn);

#endif //__DATA_READER_TEST_COMMON_HPP__
