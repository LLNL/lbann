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
//
// lbann_mild_exception .hpp - LBANN mild exception reporting macros
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MILD_EXCEPTION_HPP_INCLUDED
#define LBANN_MILD_EXCEPTION_HPP_INCLUDED

// evaluate a rare exception condition
#if 1
#define _BUILTIN_FALSE(_COND_) (__builtin_expect((_COND_),false))
#else
#define _BUILTIN_FALSE(_COND_) (_COND_)
#endif


#ifdef LBANN_DEBUG
#include "lbann/utils/exception.hpp"
//-----------------------------------------------------------------------------
#define _LBANN_DEBUG_MSG(_MSG_) \
    std::cerr << __FILE__ << " " << __LINE__ << " : " << _MSG_ << std::endl;

#define _LBANN_CRITICAL_EXCEPTION(_COND_,_MSG_,_RETVAL_) \
    if (_BUILTIN_FALSE(_COND_)) { \
      std::stringstream err; \
      err << __FILE__ << " " << __LINE__ << " : " << _MSG_ << std::endl; \
      throw lbann_exception(err.str()); \
    }

#define _LBANN_MILD_EXCEPTION(_COND_,_MSG_,_RETVAL_) \
    if (_BUILTIN_FALSE(_COND_)) { \
      _LBANN_DEBUG_MSG(_MSG_) \
      return (_RETVAL_); \
    }

#define _LBANN_SILENT_EXCEPTION(_COND_,_MSG_,_RETVAL_) \
    if (_BUILTIN_FALSE(_COND_)) { \
      return (_RETVAL_); \
    }
//-----------------------------------------------------------------------------
#else
//-----------------------------------------------------------------------------
#include <iostream>
#if 1
#define _LBANN_DEBUG_MSG(_MSG_) \
      std::cerr << __FILE__ << " " << __LINE__ << " : " << _MSG_ << std::endl;
#else
// disable message
#define _LBANN_DEBUG_MSG(_MSG_)
#endif

#define _LBANN_CRITICAL_EXCEPTION(_COND_,_MSG_,_RETVAL_) \
      if (_BUILTIN_FALSE(_COND_)) { \
        std::cerr << __FILE__ << " " << __LINE__ << " : " << _MSG_ << std::endl; \
        return (_RETVAL_); \
      }

#if 1
// print out a message and exit the current function if an exception condition is met
#define _LBANN_MILD_EXCEPTION(_COND_,_MSG_,_RETVAL_) \
      if (_BUILTIN_FALSE(_COND_)) { \
        _LBANN_DEBUG_MSG(_MSG_) \
        return (_RETVAL_); \
      }

// exit the current function if an exception condition is met
#define _LBANN_SILENT_EXCEPTION(_COND_,_MSG_,_RETVAL_) \
      if (_BUILTIN_FALSE(_COND_)) { \
        return (_RETVAL_); \
      }
#else
// skip mild exception condition checking when it is sure that it is not going to happen
#define _LBANN_MILD_EXCEPTION(_COND_,_MSG_,_RETVAL_)
#define _LBANN_SILENT_EXCEPTION(_COND_,_MSG_,_RETVAL_)
#endif
//-----------------------------------------------------------------------------
#endif

#endif // LBANN_MILD_EXCEPTION_HPP_INCLUDED
