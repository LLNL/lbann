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

#include <lbann/utils/logging.hpp>
#include "lbann/utils/exception.hpp"
#include <h2/utils/Logger.hpp>

#include <iostream>
#include <cstdlib>
#include <vector>

namespace lbann {
namespace logging {

static h2::Logger io_logger("IO");
static h2::Logger rt_logger("RT");
static h2::Logger train_logger("TRAIN");
static std::vector<h2::Logger*> logger_vec;

void setup_loggers()
{
  logger_vec.insert(logger_vec.end(), {
      &io_logger, &rt_logger, &train_logger });
  h2::setup_levels(logger_vec, "LBANN_LOG_LEVEL");
}

char const* logger_id_str(LBANN_Logger_ID id)
{
  switch (id) {
  case LBANN_Logger_ID::LOG_RT:
    return "LOG_RT";
  case LBANN_Logger_ID::LOG_IO:
    return "LOG_IO";
  case LBANN_Logger_ID::LOG_TRAIN:
    return "LOG_TRAIN";
  default:
    throw lbann_exception("Unknown LBANN_Logger_ID");
  }
}

h2::Logger& get(LBANN_Logger_ID id)
{
  switch (id) {
  case LBANN_Logger_ID::LOG_RT:
    return rt_logger;
  case LBANN_Logger_ID::LOG_IO:
    return io_logger;
  case LBANN_Logger_ID::LOG_TRAIN:
    return train_logger;
  default:
    throw lbann_exception("Unknown LBANN_Logger_ID");
  }
}

}// namespace logging
}// namespace lbann
