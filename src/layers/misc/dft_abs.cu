#include <El.hpp>

namespace lbann
{
namespace internal
{
// Have to instantiate this in device-compiled code.
void Abs(El::Matrix<El::Complex<float>, El::Device::GPU> const& in,
         El::Matrix<float, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out,
                   [] __device__ (thrust::complex<float> const& x)
                   {
                     return thrust::abs(x);
                   });
}
void Abs(El::Matrix<El::Complex<double>, El::Device::GPU> const& in,
         El::Matrix<double, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out,
                   [] __device__ (thrust::complex<double> const& x)
                   {
                     return thrust::abs(x);
                   });
}
void MyRealPart(
  El::Matrix<El::Complex<float>, El::Device::GPU> const& in,
  El::Matrix<float, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out,
                   [] __device__ (thrust::complex<float> const& x)
                   {
                     return x.real();
                   });
}
void MyRealPart(
  El::Matrix<El::Complex<double>, El::Device::GPU> const& in,
  El::Matrix<double, El::Device::GPU>& out)
{
  El::EntrywiseMap(in, out,
                   [] __device__ (thrust::complex<float> const& x)
                   {
                     return x.real();
                   });
}
void ApplyAbsGradientUpdate(
  El::Matrix<float, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<float>, El::Device::GPU>& input_output)
{
  using ComplexT = thrust::complex<float>;
  El::Combine(grad_wrt_output, input_output,
              [] __device__ (float const& dy, ComplexT const& x)
              {
                return (x == ComplexT(0.f)
                        ? ComplexT(0.f)
                        : thrust::conj(x * (dy / thrust::abs(x))));
              });
}
void ApplyAbsGradientUpdate(
  El::Matrix<double, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<double>, El::Device::GPU>& input_output)
{
  using ComplexT = thrust::complex<double>;
  El::Combine(grad_wrt_output, input_output,
              [] __device__ (double const& dy, ComplexT const& x)
              {
                return (x == ComplexT(0.0)
                        ? ComplexT(0.0)
                        : thrust::conj(x * (dy / thrust::abs(x))));
              });
}
}// namespace internal
}// namespace lbann