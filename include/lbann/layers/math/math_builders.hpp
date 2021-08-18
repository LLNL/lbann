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

#include "lbann/layers/layer.hpp"

namespace lbann
{

LBANN_DEFINE_LAYER_BUILDER(abs);
LBANN_DEFINE_LAYER_BUILDER(acos);
LBANN_DEFINE_LAYER_BUILDER(acosh);
LBANN_DEFINE_LAYER_BUILDER(add);
LBANN_DEFINE_LAYER_BUILDER(asin);
LBANN_DEFINE_LAYER_BUILDER(asinh);
LBANN_DEFINE_LAYER_BUILDER(atan);
LBANN_DEFINE_LAYER_BUILDER(atanh);
LBANN_DEFINE_LAYER_BUILDER(ceil);
LBANN_DEFINE_LAYER_BUILDER(cos);
LBANN_DEFINE_LAYER_BUILDER(cosh);
LBANN_DEFINE_LAYER_BUILDER(divide);
LBANN_DEFINE_LAYER_BUILDER(equal);
LBANN_DEFINE_LAYER_BUILDER(exp);
LBANN_DEFINE_LAYER_BUILDER(expm1);
LBANN_DEFINE_LAYER_BUILDER(floor);
LBANN_DEFINE_LAYER_BUILDER(greater);
LBANN_DEFINE_LAYER_BUILDER(greater_equal);
LBANN_DEFINE_LAYER_BUILDER(erf);
LBANN_DEFINE_LAYER_BUILDER(erfinv);
LBANN_DEFINE_LAYER_BUILDER(less);
LBANN_DEFINE_LAYER_BUILDER(less_equal);
LBANN_DEFINE_LAYER_BUILDER(log);
LBANN_DEFINE_LAYER_BUILDER(log1p);
LBANN_DEFINE_LAYER_BUILDER(logical_and);
LBANN_DEFINE_LAYER_BUILDER(logical_not);
LBANN_DEFINE_LAYER_BUILDER(logical_or);
LBANN_DEFINE_LAYER_BUILDER(logical_xor);
LBANN_DEFINE_LAYER_BUILDER(matmul);
LBANN_DEFINE_LAYER_BUILDER(max);
LBANN_DEFINE_LAYER_BUILDER(min);
LBANN_DEFINE_LAYER_BUILDER(mod);
LBANN_DEFINE_LAYER_BUILDER(multiply);
LBANN_DEFINE_LAYER_BUILDER(negative);
LBANN_DEFINE_LAYER_BUILDER(not_equal);
LBANN_DEFINE_LAYER_BUILDER(pow);
LBANN_DEFINE_LAYER_BUILDER(reciprocal);
LBANN_DEFINE_LAYER_BUILDER(round);
LBANN_DEFINE_LAYER_BUILDER(rsqrt);
LBANN_DEFINE_LAYER_BUILDER(safe_divide);
LBANN_DEFINE_LAYER_BUILDER(safe_reciprocal);
LBANN_DEFINE_LAYER_BUILDER(sign);
LBANN_DEFINE_LAYER_BUILDER(sin);
LBANN_DEFINE_LAYER_BUILDER(sinh);
LBANN_DEFINE_LAYER_BUILDER(sqrt);
LBANN_DEFINE_LAYER_BUILDER(square);
LBANN_DEFINE_LAYER_BUILDER(squared_difference);
LBANN_DEFINE_LAYER_BUILDER(subtract);
LBANN_DEFINE_LAYER_BUILDER(tan);
LBANN_DEFINE_LAYER_BUILDER(tanh);

}// namespace lbann
