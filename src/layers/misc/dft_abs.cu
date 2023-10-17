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
#include <El.hpp>

namespace lbann {
namespace internal {
// Have to instantiate this in device-compiled code.
void Abs(El::Matrix<El::Complex<float>, El::Device::GPU> const& in,
         El::Matrix<float, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out, [] __device__(thrust::complex<float> const& x) {
    return thrust::abs(x);
  });
}
#ifdef LBANN_HAS_DOUBLE
void Abs(El::Matrix<El::Complex<double>, El::Device::GPU> const& in,
         El::Matrix<double, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out, [] __device__(thrust::complex<double> const& x) {
    return thrust::abs(x);
  });
}
#endif // LBANN_HAS_DOUBLE
void MyRealPart(El::Matrix<El::Complex<float>, El::Device::GPU> const& in,
                El::Matrix<float, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out, [] __device__(thrust::complex<float> const& x) {
    return x.real();
  });
}
#ifdef LBANN_HAS_DOUBLE
void MyRealPart(El::Matrix<El::Complex<double>, El::Device::GPU> const& in,
                El::Matrix<double, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out, [] __device__(thrust::complex<float> const& x) {
    return x.real();
  });
}
#endif // LBANN_HAS_DOUBLE
void ApplyAbsGradientUpdate(
  El::Matrix<float, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<float>, El::Device::GPU>& input_output)
{
  using ComplexT = thrust::complex<float>;
  El::Combine(grad_wrt_output,
              input_output,
              [] __device__(float const& dy, ComplexT const& x) {
                return (x == ComplexT(0.f)
                          ? ComplexT(0.f)
                          : thrust::conj(x * (dy / thrust::abs(x))));
              });
}
#ifdef LBANN_HAS_DOUBLE
void ApplyAbsGradientUpdate(
  El::Matrix<double, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<double>, El::Device::GPU>& input_output)
{
  using ComplexT = thrust::complex<double>;
  El::Combine(grad_wrt_output,
              input_output,
              [] __device__(double const& dy, ComplexT const& x) {
                return (x == ComplexT(0.0)
                          ? ComplexT(0.0)
                          : thrust::conj(x * (dy / thrust::abs(x))));
              });
}
#endif // LBANN_HAS_DOUBLE
} // namespace internal
} // namespace lbann
