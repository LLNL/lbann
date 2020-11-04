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

#include <lbann/utils/rocfft_wrapper.hpp>
#include <lbann/utils/exception.hpp>

#include <string>

namespace lbann
{
namespace rocfft
{

std::string value_as_string(rocfftResult_t r)
{
#define ROCFFT_VALUE_AS_STRING_CASE(VAL) \
  case VAL: return #VAL
  switch (r) {
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_SUCCESS       );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INVALID_PLAN  );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_ALLOC_FAILED  );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INVALID_TYPE  );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INVALID_VALUE );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INTERNAL_ERROR);
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_EXEC_FAILED   );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_SETUP_FAILED  );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INVALID_SIZE  );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_UNALIGNED_DATA);
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INCOMPLETE_PARAMETER_LIST);
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_INVALID_DEVICE);
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_PARSE_ERROR   );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_NO_WORKSPACE  );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_NOT_IMPLEMENTED);
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_LICENSE_ERROR );
    ROCFFT_VALUE_AS_STRING_CASE(ROCFFT_NOT_SUPPORTED );
  default:
    LBANN_ERROR("Unknown cufftResult_t value.");
  }
  return "";
#undef ROCFFT_VALUE_AS_STRING_CASE
}

std::string result_string(rocfftResult_t r)
{
  switch (r) {
  case ROCFFT_SUCCESS       :
    return "The cuFFT operation was successful";
  case ROCFFT_INVALID_PLAN  :
    return "cuFFT was passed an invalid plan handle";
  case ROCFFT_ALLOC_FAILED  :
    return "cuFFT failed to allocate GPU or CPU memory";
  case ROCFFT_INVALID_TYPE  :
    return "No longer used (value=ROCFFT_INVALID_TYPE)";
  case ROCFFT_INVALID_VALUE :
    return "User specified an invalid pointer or parameter";
  case ROCFFT_INTERNAL_ERROR:
    return "Driver or internal cuFFT library error";
  case ROCFFT_EXEC_FAILED   :
    return "Failed to execute an FFT on the GPU";
  case ROCFFT_SETUP_FAILED  :
    return "The cuFFT library failed to initialize";
  case ROCFFT_INVALID_SIZE  :
    return "User specified an invalid transform size";
  case ROCFFT_UNALIGNED_DATA:
    return "No longer used (value=ROCFFT_UNALIGNED_DATA)";
  case ROCFFT_INCOMPLETE_PARAMETER_LIST:
    return "Missing parameters in call";
  case ROCFFT_INVALID_DEVICE:
    return "Execution of a plan was on different GPU than plan creation";
  case ROCFFT_PARSE_ERROR   :
    return "Internal plan database error";
  case ROCFFT_NO_WORKSPACE  :
    return "No workspace has been provided prior to plan execution";
  case ROCFFT_NOT_IMPLEMENTED:
    return "Function does not implement functionality for parameters given.";
  case ROCFFT_LICENSE_ERROR :
    return "Used in previous versions.";
  case ROCFFT_NOT_SUPPORTED :
    return "Operation is not supported for parameters given.";
  default:
    LBANN_ERROR("Unknown cufftResult_t value.");
  }
  return "";
}
}// namespace rocfft
}// namespace lbann
