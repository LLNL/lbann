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

#include "lbann/utils/number_theory.hpp"
#include "lbann/utils/exception.hpp"
#include <algorithm>
#include <unordered_map>

namespace lbann {
namespace number_theory {

int prime(int n) {
  if (n < 0) {
    std::stringstream err;
    err << "invalid index (" << n << ")";
    LBANN_ERROR(err.str());
  }
  if (n == 0) { return 2; }

  // Expand list of odd primes if needed
  // Note: Odd primes are cached for future function calls. We iterate
  // through odd numbers and check whether they have any prime
  // divisors. The smallest prime divisor of a composite number q must
  // be less than or equal to sqrt(q).
  static std::vector<int> cache = {3};
  for (int q = cache.back() + 2;
       n-1 >= (int) cache.size();
       q += 2) {
    for (const auto& p : cache) {
      if (q % p == 0) { break; }
      if (p * p > q) {
        cache.push_back(q);
        break;
      }
    }
  }

  // Return cached prime
  return cache[n-1];

}

std::vector<int> prime_factors(int n) {
  if (n < 2) {
    std::stringstream err;
    err << "invalid number to factorize (" << n << ")";
    LBANN_ERROR(err.str());
  }

  // Extract primes from n
  // Note: If n is greater than one and has no prime divisors less
  // than or equal to sqrt(n), it is prime.
  std::vector<int> factors;
  for (int i = 0, p = prime(i); p * p <= n; p = prime(++i)) {
    while (n % p == 0) {
      factors.push_back(p);
      n /= p;
    }
  }
  if (n > 1) { factors.push_back(n); }

  return factors;
}

std::vector<int> balanced_factors(int n, int num_factors) {
  std::stringstream err;
  if (n < 1) {
    err << "invalid number to factorize (" << n << ")";
    LBANN_ERROR(err.str());
  }
  if (num_factors < 1) {
    err << "invalid number of factors (" << num_factors << ")";
    LBANN_ERROR(err.str());
  }

  // Trivial case when n = 1
  if (n == 1) { return std::vector<int>(num_factors, 1); };

  // Get prime factorization
  const auto& primes = prime_factors(n);

  // Compute balanced factors
  /// @todo Non-greedy algorithm
  std::vector<int> factors(num_factors, 1);
  for (int i = primes.size() - 1; i >= 0; --i) {
    factors.front() *= primes[i];
    std::inplace_merge(factors.begin(),
                       factors.begin()+1,
                       factors.end());
  }
  return factors;

}

} // namespace number_theory
} // lbann
