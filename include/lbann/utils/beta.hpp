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

#ifndef LBANN_UTILS_BETA_HPP
#define LBANN_UTILS_BETA_HPP

#include <random>
#include <ostream>
#include <istream>
#include <cmath>

#include "lbann/utils/random.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/**
 * Produces random floating point values drawn from a Beta distribution with
 * parameters a > 0 and b > 0.
 *
 * See:
 *
 *     https://en.wikipedia.org/wiki/Beta_distribution
 *
 * for more details.
 */
template <typename RealType = double>
class beta_distribution {
public:
  using result_type = RealType;

  class param_type {
  public:
    using distribution_type = beta_distribution;

    explicit param_type(RealType a, RealType b) :
      m_a(a), m_b(b) {
      if (a <= RealType(0) || b <= RealType(0)) {
        LBANN_ERROR("Beta distribution parameters must be positive");
      }
    }

    constexpr RealType a() const { return m_a; }
    constexpr RealType b() const { return m_b; }

    bool operator==(const param_type& other) const {
      return m_a == other.m_a && m_b == other.m_b;
    }
    bool operator!=(const param_type& other) const {
      return m_a != other.m_a || m_b != other.m_b;
    }
  private:
    RealType m_a, m_b;
  };

  explicit beta_distribution(RealType a, RealType b) :
    m_params(a, b), m_gamma_a(a), m_gamma_b(b) {}
  explicit beta_distribution(const param_type& p) :
    m_params(p), m_gamma_a(p.a()), m_gamma_b(p.b()) {}

  result_type a() const { return m_params.a(); }
  result_type b() const { return m_params.b(); }

  void reset() {}

  param_type param() const { return m_params; }
  void param(const param_type& p) {
    m_params = p;
    m_gamma_a = gamma_dist(p.a());
    m_gamma_b = gamma_dist(p.b());
  }

  template <typename Generator>
  result_type operator()(Generator& g) {
    return generate(g);
  }
  template <typename Generator>
  result_type operator()(Generator& g, const param_type& p) {
    return generate(g, p);
  }

  result_type min() const { return result_type(0); }
  result_type max() const { return result_type(1); }

  bool operator==(const beta_distribution<result_type>& other) const {
    return param() == other.param();
  }
  bool operator!=(const beta_distribution<result_type>& other) const {
    return param() != other.param();
  }

private:
  param_type m_params;

  using gamma_dist = std::gamma_distribution<RealType>;
  gamma_dist m_gamma_a, m_gamma_b;

  // Generator for when we use the distribution's parameters.
  template <typename Generator>
  result_type generate(Generator& g) {
    if (a() <= result_type(1) && b() <= result_type(1)) {
      return generate_johnk(g, m_params.a(), m_params.b());
    } else {
      return generate_gamma(g, m_gamma_a, m_gamma_b);
    }
  }
  // Generator for when we use specified parameters.
  template <typename Generator>
  result_type generate(Generator& g, const param_type& p) {
    if (p.a() <= result_type(1) && p.b() <= result_type(1)) {
      return generate_johnk(g, p.a(), p.b());
    } else {
      gamma_dist gamma_a(p.a()), gamma_b(p.b());
      return generate_gamma(g, gamma_a, gamma_b);
    }
  }

  /**
   * Generate Beta-distributed values using Johnk's algorithm.
   * This is a rejection-sampling algorithm that only needs a few
   * uniformly random values.
   * 
   * See:
   *
   *     Johnk, H. D. "Erzeugung von betaverteilten und gammaverteilten
   *     Zufallszahlen." Metrika 8, no. 1 (1964).
   *
   * For an English-language presentation, see:
   *
   *     Atkinson, A. C. and M. C. Pearce. "The computer generation of beta,
   *     gamma and normal random variables." Journal of the Royal Statistical
   *     Society: Series A (General) 139, no. 4 (1976).
   *
   * This includes fixes for numerical stability when the parameters are small,
   * see:
   *
   *     https://github.com/numpy/numpy/issues/5851
   *
   * for discussion there; and a catch for the (extremely rare) case of the RNG
   * giving us U and V both exactly 0.
   *
   * Note: There should be an umlaut on the "o" in "Johnk", but blame poor
   * unicode support.
   */
  template <typename Generator>
  result_type generate_johnk(Generator& g, result_type a, result_type b) {
    while (true) {
      const result_type U = fast_random_uniform<result_type>(g);
      const result_type V = fast_random_uniform<result_type>(g);
      const result_type X = std::pow(U, result_type(1) / a);
      const result_type Y = std::pow(V, result_type(1) / b);
      const result_type XplusY = X + Y;
      if (XplusY <= result_type(1.0)) {
        if (XplusY > result_type(0)) {
          return X / XplusY;
        } else if (U != result_type(0) && V != result_type(0)) {
          // Work with logs instead if a/b is too small.
          result_type logX = std::log(U) / a;
          result_type logY = std::log(V) / b;
          const result_type log_max = std::max(logX, logY);
          logX -= log_max;
          logY -= log_max;
          return std::exp(logX - std::log(std::exp(logX) + std::exp(logY)));
        }
      }
    }
  }

  /**
   * Generate Beta-distributed values based on Gamma distributions.
   * See:
   *     https://en.wikipedia.org/wiki/Beta_distribution#Generating_beta-distributed_random_variates
   * for details.
   */
  template <typename Generator>
  result_type generate_gamma(Generator& g, gamma_dist& gamma_a,
                             gamma_dist& gamma_b) {
    const result_type Ga = gamma_a(g);
    const result_type Gb = gamma_b(g);
    return Ga / (Ga + Gb);
  }
};

template <typename CharT, typename RealType>
std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT>& os,
                                      const beta_distribution<RealType>& d) {
  os << "~Beta(" << d.a() << "," << d.b() << ")";
  return os;
}

template <typename CharT, typename RealType>
std::basic_istream<CharT>& operator>>(std::basic_istream<CharT>& is,
                                      beta_distribution<RealType>& d) {
  std::string s;
  RealType a, b;
  if (std::getline(is, s, '(') && s == "~Beta"
      && is >> a
      && is.get() == ','
      && is >> b
      && is.get() == ')') {
    d = beta_distribution<RealType>(a, b);
  } else {
    is.setstate(std::ios::failbit);
  }
  return is;
}

}  // namespace lbann

#endif  // LBANN_UTILS_BETA_HPP
