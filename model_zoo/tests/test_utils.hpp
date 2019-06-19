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
// http://github.com/LBANN.
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
// test_utils.hpp - Utilities for doing testing
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_TEST_UTILS_HPP_INCLUDED
#define LBANN_TEST_UTILS_HPP_INCLUDED

#include <vector>

template <typename T, typename U>
inline void assert_eq(T x, U y, const char *xname, const char *yname,
                      const char *file, size_t line) {
  if (x != y) {
    std::cout << "ASSERT EQ failure: " << xname << " (" << x <<
              ") != " << yname << " (" << y << ") at " << file << ":" << line <<
              std::endl;
    exit(1);
  }
}

template <typename T, typename U>
inline void assert_neq(T x, U y, const char *xname, const char *yname,
                       const char *file, size_t line) {
  if (x == y) {
    std::cout << "ASSERT NEQ failure: " << xname << " (" << x <<
              ") == " << yname << " (" << y << ") at " << file << ":" << line <<
              std::endl;
    exit(1);
  }
}

inline void assert_true(bool x, const char *xname, const char *file,
                        size_t line) {
  if (!x) {
    std::cout << "ASSERT TRUE failure: " << xname << " (" << x << ") at " <<
              file << ":" << line << std::endl;
    exit(1);
  }
}

inline void assert_false(bool x, const char *xname, const char *file,
                         size_t line) {
  if (x) {
    std::cout << "ASSERT FALSE failure: " << xname << " (" << x << ") at " <<
              file << ":" << line << std::endl;
    exit(1);
  }
}

inline void assert_mat_eq(const lbann::Mat& x, const lbann::Mat& y, lbann::DataType tol,
                          const char *xname, const char *yname,
                          const char *file, size_t line) {
  // There are better ways to compare floating point values, but this should be
  // sufficient for now. Google Test has a very comprehensive and well-tested
  // floating-point almost-equal method if needed.
  assert_eq(x.Height(), y.Height(), xname, yname, file, line);
  assert_eq(x.Width(), y.Width(), xname, yname, file, line);
  for (int i = 0; i < x.Height(); ++i) {
    for (int j = 0; j < x.Width(); ++j) {
      if (fabs(x.Get(i, j) - y.Get(i, j)) > tol) {
        std::cout << "ASSERT_MAT_EQ failure: " << xname << "(" << i << ", " <<
                  j << ") [" << x.Get(i, j) << "] != " << yname << "(" << i << ", " <<
                  j << ") [" << y.Get(i, j) << "] at " << file << ":" << line <<
                  std::endl;
        exit(1);
      }
    }
  }
}

inline void assert_mat_eq(const lbann::DistMat& x, const lbann::DistMat& y, lbann::DataType tol,
                          const char *xname, const char *yname,
                          const char *file, size_t line) {
  assert_mat_eq(x.LockedMatrix(), y.LockedMatrix(), tol, xname, yname, file,
                line);
}

inline void assert_mat_neq(const lbann::Mat& x, const lbann::Mat& y, lbann::DataType tol,
                           const char *xname, const char *yname,
                           const char *file, size_t line) {
  // Still ensure these matrices are the same size.
  assert_eq(x.Height(), y.Height(), xname, yname, file, line);
  assert_eq(x.Width(), y.Width(), xname, yname, file, line);
  for (int i = 0; i < x.Height(); ++i) {
    for (int j = 0; j < x.Width(); ++j) {
      if (fabs(x.Get(i, j) - y.Get(i, j)) > tol) {
        return;  // Matrices differ.
      }
    }
  }
  std::cout << "ASSERT_MAT_NEQ failure: " << xname << " == " << yname <<
            " at " << file << ":" << line << std::endl;
  exit(1);
}

inline void assert_mat_neq(const lbann::DistMat& x, const lbann::DistMat& y, lbann::DataType tol,
                           const char *xname, const char *yname,
                           const char *file, size_t line) {
  assert_mat_neq(x.LockedMatrix(), y.LockedMatrix(), tol, xname, yname, file,
                 line);
}

template <typename T>
inline void assert_vector_eq(const std::vector<T> x, const std::vector<T> y,
                             const char *xname, const char *yname,
                             const char *file, size_t line) {
  assert_eq(x.size(), y.size(), xname, yname, file, line);
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != y[i]) {
      std::cout << "ASSERT_VECTOR_EQ failure: " << xname << "[" << i <<
                "] != " << yname << "[" << i << "] at " << file << ":" << line <<
                std::endl;
      exit(1);
    }
  }
}

template <typename T>
inline void assert_vector_neq(const std::vector<T> x, const std::vector<T> y,
                              const char *xname, const char *yname,
                              const char *file, size_t line) {
  assert_eq(x.size(), y.size(), xname, yname, file, line);
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != y[i]) {
      return;
    }
  }
  std::cout << "ASSERT_VECTOR_NEQ failure: " << xname << " == " << yname <<
            " at " << file << ":" << line << std::endl;
}

#define ASSERT_EQ(x, y) assert_eq(x, y, #x, #y, __FILE__, __LINE__)
#define ASSERT_NEQ(x, y) assert_neq(x, y, #x, #y, __FILE__, __LINE__)
#define ASSERT_TRUE(x) assert_true(x, #x, __FILE__, __LINE__)
#define ASSERT_FALSE(x) assert_false(x, #x, __FILE__, __LINE__)
#define ASSERT_MAT_EQ_TOL(x, y, tol) assert_mat_eq(x, y, tol, #x, #y, __FILE__, __LINE__)
#define ASSERT_MAT_EQ(x, y) assert_mat_eq(x, y, 1e-4, #x, #y, __FILE__, __LINE__)
#define ASSERT_MAT_NEQ_TOL(x, y, tol) assert_mat_neq(x, y, tol, #x, #y, __FILE__, __LINE__)
#define ASSERT_MAT_NEQ(x, y) assert_mat_neq(x, y, 1e-4, #x, #y, __FILE__, __LINE__)
#define ASSERT_VECTOR_EQ(x, y) assert_vector_eq(x, y, #x, #y, __FILE__, __LINE__)
#define ASSERT_VECTOR_NEQ(x, y) assert_vector_neq(x, y, #x, #y, __FILE__, __LINE__)

/**
 * Compute the absolute error between approx_val and true_val, both overall
 * (in the return value) and element-wise (in elemerr).
 */
lbann::DataType absolute_error(lbann::CPUMat& approx_val, lbann::CPUMat& true_val, lbann::CPUMat& elemerr) {
  ASSERT_EQ(approx_val.Width(), true_val.Width());
  ASSERT_EQ(approx_val.Height(), true_val.Height());
  elemerr = true_val;
  El::Axpy(lbann::DataType{-1}, approx_val, elemerr);
  lbann::DataType abs_err = El::EntrywiseNorm(elemerr, 1);
  El::EntrywiseMap(elemerr, std::function<lbann::DataType(const lbann::DataType&)>(
  [](const lbann::DataType& x) {
    return fabs(x);
  }));
  return abs_err;
}

/**
 * Compute the relative error between approx_val and true_val, both overall
 * (in the return value) and element-wise (in elemerr).
 */
lbann::DataType relative_error(lbann::CPUMat& approx_val, lbann::CPUMat& true_val, lbann::CPUMat& elemerr) {
  ASSERT_EQ(approx_val.Width(), true_val.Width());
  ASSERT_EQ(approx_val.Height(), true_val.Height());
  lbann::DataType abs_err = absolute_error(approx_val, true_val, elemerr);
  lbann::DataType rel_err = abs_err / El::EntrywiseNorm(true_val, 1);
  lbann::Mat true_copy(true_val);
  El::EntrywiseMap(true_copy, std::function<lbann::DataType(const lbann::DataType&)>(
  [](const lbann::DataType& x) {
    return 1.0f / fabs(x);
  }));
  lbann::Mat elemerr_copy(elemerr);
  El::Hadamard(static_cast<lbann::AbsMat&>(elemerr_copy), static_cast<lbann::AbsMat&>(true_copy), static_cast<lbann::AbsMat&>(elemerr));
  return rel_err;
}

#endif  // LBANN_TEST_UTILS_HPP_INCLUDED
