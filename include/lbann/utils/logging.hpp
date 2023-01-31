////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LOGGING_HPP_INCLUDED
#define LBANN_LOGGING_HPP_INCLUDED

#include <h2/utils/Logger.hpp>

#include <iostream>
#include <cstdlib>
#include <vector>

namespace lbann {

static h2::Logger logger("test_logger");
static std::vector<h2::Logger*> logger_vec;

static inline void setup_loggers()
{
  //putenv("TEST_LOG_MASK=trace|debug|info|warn|error|critical, io=debug|info|warn, training=critical|error, test_logger=debug|info|warn");
  //putenv("TEST_LOG_LEVEL=critical, io=debug");
  logger_vec.push_back(&logger);
  h2::setup_levels(logger_vec, "TEST_LOG_LEVEL");
  h2::setup_masks(logger_vec, "TEST_LOG_MASK");
}
}

/*
#define LBANN_RT_TRACE(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::TRACE))                  \
    logger.get().trace(__VA_ARGS__);                                       \

#define LBANN_RT_DEBUG(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::DEBUG))                  \
    logger.get().debug(__VA_ARGS__);                                       \

#define LBANN_RT_INFO(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::INFO))                   \
    logger.get().info(__VA_ARGS__);                                        \

#define LBANN_RT_WARN(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::WARN))                   \
    logger.get().warn(__VA_ARGS__);                                        \

#define LBANN_RT_ERROR(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::ERROR))                  \
    logger.get().error(__VA_ARGS__);                                       \

#define LBANN_RT_CRITICAL(...)                                             \
  if (logger.should_log(h2::Logger::LogLevelType::CRITICAL))               \
    logger.get().critical(__VA_ARGS__);                                    \

#define LBANN_IO_TRACE(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::TRACE))                  \
    logger.get().trace(__VA_ARGS__);                                       \

#define LBANN_IO_DEBUG(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::DEBUG))                  \
    logger.get().debug(__VA_ARGS__);                                       \

#define LBANN_IO_INFO(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::INFO))                   \
    logger.get().info(__VA_ARGS__);                                        \

#define LBANN_IO_WARN(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::WARN))                   \
    logger.get().warn(__VA_ARGS__);                                        \

#define LBANN_IO_ERROR(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::ERROR))                  \
    logger.get().error(__VA_ARGS__);                                       \

#define LBANN_IO_CRITICAL(...)                                             \
  if (logger.should_log(h2::Logger::LogLevelType::CRITICAL))               \
    logger.get().critical(__VA_ARGS__);                                    \

#define LBANN_TRAINING_TRACE(...)                                          \
  if (logger.should_log(h2::Logger::LogLevelType::TRACE))                  \
    logger.get().trace(__VA_ARGS__);                                       \

#define LBANN_TRAINING_DEBUG(...)                                          \
  if (logger.should_log(h2::Logger::LogLevelType::DEBUG))                  \
    logger.get().debug(__VA_ARGS__);                                       \

#define LBANN_TRAINING_INFO(...)                                           \
  if (logger.should_log(h2::Logger::LogLevelType::INFO))                   \
    logger.get().info(__VA_ARGS__);                                        \

#define LBANN_TRAINING_WARN(...)                                           \
  if (logger.should_log(h2::Logger::LogLevelType::WARN))                   \
    logger.get().warn(__VA_ARGS__);                                        \

#define LBANN_TRAINING_ERROR(...)                                          \
  if (logger.should_log(h2::Logger::LogLevelType::ERROR))                  \
    logger.get().error(__VA_ARGS__);                                       \

#define LBANN_TRAINING_CRITICAL(...)                                       \
  if (logger.should_log(h2::Logger::LogLevelType::CRITICAL))               \
    logger.get().critical(__VA_ARGS__);                                    \
*/
#endif // LBANN_LOGGING_HPP_INCLUDED
