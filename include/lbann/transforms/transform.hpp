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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_TRANSFORMS_TRANSFORM_HPP_INCLUDED
#define LBANN_TRANSFORMS_TRANSFORM_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/type_erased_matrix.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {
namespace transform {

/**
 * Abstract base class for transforms on data.
 * 
 * A transform takes a CPUMat and modifies it in-place. Transforms should
 * be thread-safe, as one instance of a transform may be called concurrently
 * within multiple threads.
 *
 * Because transforms may switch between underlying data types throughout the
 * pipeline, everything is done in terms of a type_erased_matrix, which can
 * swap between underlying data types.
 */
class transform {
public:
  transform() = default;
  transform(const transform&) = default;
  transform& operator=(const transform&) = default;
  virtual ~transform() = default;

  /** Create a copy of the transform instance. */
  virtual transform* copy() const = 0;

  /** Human-readable type name. */
  virtual std::string get_type() const = 0;
  /** Human-readable description. */
  virtual description get_description() const {
    return description(get_type() + " transform");
  }

  /** True if the transform supports non-in-place apply. */
  virtual bool supports_non_inplace() const {
    return false;
  }

  /**
   * Apply the transform to data.
   * @param data The input data to transform, which is modified in-place. The
   *   matrix shuold be contiguous.
   * @param dims The dimensions of the data tensor. For "plain data", dims
   * should have one entry, giving its size. For images, dims should have three
   * entries: channels, height, width.
   * @note dims is a hack until we have proper tensors.
   */
  virtual void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) = 0;

  /**
   * Apply the transform to data.
   * This does not modify data in-place but places its output in out.
   */
  virtual void apply(utils::type_erased_matrix& data, CPUMat& out,
                     std::vector<size_t>& dims) {
    LBANN_ERROR("Non-in-place apply not implemented.");
  }
protected:
  /** Return a value uniformly at random in [a, b). */
  static inline float get_uniform_random(float a, float b) {
    fast_rng_gen& gen = get_fast_io_generator();
    std::uniform_real_distribution<float> dist(a, b);
    return dist(gen);
  }
  /** Return true with probability p. */
  static inline bool get_bool_random(float p) {
    return get_uniform_random(0.0, 1.0) < p;
  }
  /** Return an integer uniformly at random in [a, b). */
  static inline El::Int get_uniform_random_int(El::Int a, El::Int b) {
    fast_rng_gen& gen = get_fast_io_generator();
    return fast_rand_int(gen, b - a) + a;
  }
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_TRANSFORM_HPP_INCLUDED
