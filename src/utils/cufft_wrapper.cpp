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

#include <lbann/utils/cufft_wrapper.hpp>
#include <lbann/utils/exception.hpp>

#include <string>

namespace lbann
{
namespace cufft
{

std::string value_as_string(cufftResult_t r)
{
#define CUFFT_VALUE_AS_STRING_CASE(VAL) \
  case VAL: return #VAL
  switch (r) {
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_SUCCESS       );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INVALID_PLAN  );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_ALLOC_FAILED  );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INVALID_TYPE  );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INVALID_VALUE );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INTERNAL_ERROR);
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_EXEC_FAILED   );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_SETUP_FAILED  );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INVALID_SIZE  );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_UNALIGNED_DATA);
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INCOMPLETE_PARAMETER_LIST);
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_INVALID_DEVICE);
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_PARSE_ERROR   );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_NO_WORKSPACE  );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_NOT_IMPLEMENTED);
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_LICENSE_ERROR );
    CUFFT_VALUE_AS_STRING_CASE(CUFFT_NOT_SUPPORTED );
  default:
    LBANN_ERROR("Unknown cufftResult_t value.");
  }
  return "";
#undef CUFFT_VALUE_AS_STRING_CASE
}

std::string result_string(cufftResult_t r)
{
  switch (r) {
  case CUFFT_SUCCESS       :
    return "The cuFFT operation was successful";
  case CUFFT_INVALID_PLAN  :
    return "cuFFT was passed an invalid plan handle";
  case CUFFT_ALLOC_FAILED  :
    return "cuFFT failed to allocate GPU or CPU memory";
  case CUFFT_INVALID_TYPE  :
    return "No longer used (value=CUFFT_INVALID_TYPE)";
  case CUFFT_INVALID_VALUE :
    return "User specified an invalid pointer or parameter";
  case CUFFT_INTERNAL_ERROR:
    return "Driver or internal cuFFT library error";
  case CUFFT_EXEC_FAILED   :
    return "Failed to execute an FFT on the GPU";
  case CUFFT_SETUP_FAILED  :
    return "The cuFFT library failed to initialize";
  case CUFFT_INVALID_SIZE  :
    return "User specified an invalid transform size";
  case CUFFT_UNALIGNED_DATA:
    return "No longer used (value=CUFFT_UNALIGNED_DATA)";
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "Missing parameters in call";
  case CUFFT_INVALID_DEVICE:
    return "Execution of a plan was on different GPU than plan creation";
  case CUFFT_PARSE_ERROR   :
    return "Internal plan database error";
  case CUFFT_NO_WORKSPACE  :
    return "No workspace has been provided prior to plan execution";
  case CUFFT_NOT_IMPLEMENTED:
    return "Function does not implement functionality for parameters given.";
  case CUFFT_LICENSE_ERROR :
    return "Used in previous versions.";
  case CUFFT_NOT_SUPPORTED :
    return "Operation is not supported for parameters given.";
  default:
    LBANN_ERROR("Unknown cufftResult_t value.");
  }
  return "";
}
}// namespace cufft
}// namespace lbann
