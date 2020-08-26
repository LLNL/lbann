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
}
void ApplyAbsGradientUpdate(
  El::Matrix<double, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<double>, El::Device::GPU>& input_output)
{
}
}// namespace internal
}// namespace lbann