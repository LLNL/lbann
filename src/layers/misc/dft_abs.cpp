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

#include <lbann/layers/misc/dft_abs.hpp>
#include <lbann/utils/memory.hpp>

// This file is only compiled if LBANN has FFTW support, so we don't
// need to check for it here.

#include <fftw3.h>
#include <lbann/utils/fftw_wrapper.hpp>

#if defined LBANN_HAS_CUDA
#include <cufft.h>
#include <lbann/utils/cufft_wrapper.hpp>
#endif // defined LBANN_HAS_CUDA

namespace {
// Metaprogramming for the layer impl.
template <typename T, El::Device D>
struct FFTBackendT;

template <typename T, El::Device D>
using FFTBackend = typename FFTBackendT<T, D>::type;

template <typename T>
struct FFTBackendT<T, El::Device::CPU>
{
  using type = lbann::fftw::FFTWWrapper<T>;
};

using namespace El;
// Updates B(i,j) = func(A(i,j), B(i,j));
template <typename S, typename T, typename F>
void Combine(Matrix<S, Device::CPU> const& A, Matrix<T, Device::CPU>& B, F func)
{
  EL_DEBUG_CSE;
  const Int m = A.Height();
  const Int n = A.Width();

  if (B.Height() != m || B.Width() != n)
    LBANN_ERROR("B must be the same size as A.");

  S const* ABuf = A.LockedBuffer();
  T* BBuf = B.Buffer();
  Int const ALDim = A.LDim();
  Int const BLDim = B.LDim();

  // Use entry-wise parallelization for column vectors. Otherwise
  // use column-wise parallelization.
  if (n == 1) {
    EL_PARALLEL_FOR
    for (Int i = 0; i < m; ++i) {
      BBuf[i] = func(ABuf[i], BBuf[i]);
    }
  }
  else {
    EL_PARALLEL_FOR_COLLAPSE2
    for (Int j = 0; j < n; ++j) {
      for (Int i = 0; i < m; ++i) {
        BBuf[i + j * BLDim] = func(ABuf[i + j * ALDim], BBuf[i + j * BLDim]);
      }
    }
  }
}

#ifdef LBANN_HAS_GPU

template <typename T>
struct FFTBackendT<T, El::Device::GPU>
{
  using type = lbann::cufft::cuFFTWrapper<T>;
};

#endif // LBANN_HAS_GPU

} // namespace

namespace lbann {
namespace internal {

template <typename T>
static void Abs(El::Matrix<El::Complex<T>, El::Device::CPU> const& in,
                El::Matrix<T, El::Device::CPU>& out)
{
  using ComplexT = El::Complex<T>;
  El::EntrywiseMap(in,
                   out,
                   std::function<T(ComplexT const&)>(
                     [](ComplexT const& x) { return El::Abs(x); }));
}

template <typename T>
static void ApplyAbsGradientUpdate(
  El::Matrix<T, El::Device::CPU> const& grad_wrt_output,
  El::Matrix<El::Complex<T>, El::Device::CPU>& input_output)
{
  using ComplexT = El::Complex<T>;
  Combine(grad_wrt_output, input_output, [](T const& dy, ComplexT const& x) {
    return (x == ComplexT(T(0)) ? ComplexT(T(0))
                                : El::Conj(x * (dy / El::Abs(x))));
  });
}

template <typename T>
static void MyRealPart(El::Matrix<El::Complex<T>, El::Device::CPU> const& in,
                       El::Matrix<T, El::Device::CPU>& out)
{
  // out should be setup, so make sure no resize will happen.
  if ((in.Height() != out.Height()) || (in.Width() != out.Width()))
    LBANN_ERROR("Incompatible matrix dimensions.");
  El::RealPart(in, out);
}

#ifdef LBANN_HAS_GPU
// Have to instantiate this in device-compiled code.
void Abs(El::Matrix<El::Complex<float>, El::Device::GPU> const& in,
         El::Matrix<float, El::Device::GPU>& out);
void ApplyAbsGradientUpdate(
  El::Matrix<float, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<float>, El::Device::GPU>& input_output);
void MyRealPart(El::Matrix<El::Complex<float>, El::Device::GPU> const& in,
                El::Matrix<float, El::Device::GPU>& out);
#ifdef LBANN_HAS_DOUBLE
void Abs(El::Matrix<El::Complex<double>, El::Device::GPU> const& in,
         El::Matrix<double, El::Device::GPU>& out);
void ApplyAbsGradientUpdate(
  El::Matrix<double, El::Device::GPU> const& grad_wrt_output,
  El::Matrix<El::Complex<double>, El::Device::GPU>& input_output);
void MyRealPart(El::Matrix<El::Complex<double>, El::Device::GPU> const& in,
                El::Matrix<double, El::Device::GPU>& out);
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_GPU

} // namespace internal

// NB (trb 08/04/2020): This is only exposing FFTW and cuFFT for
// now. ROCm support via rocFFT will come later, as might (?)
// oneAPI/MKL support.

template <typename T, El::Device D>
class dft_abs_impl
{
  static_assert(El::IsReal<T>::value,
                "dft_abs_layer only supports real input/output right now.");

  using ComplexT = El::Complex<T>;
  using RealMatType = El::Matrix<T, D>;
  using ComplexMatType = El::Matrix<ComplexT, D>;

public:
  /** @brief Constructor.
   *  @param dims The dimensions corresponding to a full sample.
   *  @note The minibatch dimension is not needed until setup and may
   *        change, while we assume the sample dims stay constant and
   *        are the same for each sample.
   */
  dft_abs_impl(std::vector<int> const& dims) : full_dims_{dims} {}

  ~dft_abs_impl() {}

  dft_abs_impl(dft_abs_impl&&) noexcept = default;

  dft_abs_impl(dft_abs_impl const& other) : dft_abs_impl(other.full_dims_) {}

  /** @brief Compute the action of forward propagation.
   *  @details This is a DFT followed by and entrywise Abs() application.
   *  @param input The (real) input to the DFT.
   *  @param output The (real) output of the Abs() operation.
   */
  void do_fp_compute(RealMatType const& input, RealMatType& output) const
  {
    // Copy to complex values
    El::Copy(input, workspace_);

    // This does the first part: the DFT
    fft_impl_.compute_forward(workspace_);

    // Now output the absolute value -- keep it real!
    internal::Abs(workspace_, output);
  }

  void do_bp_compute(RealMatType const& local_grad_wrt_output,
                     RealMatType& local_grad_wrt_input) const
  {
    // Similarly, back-prop happens in two parts. The first stage
    // applies the gradient of the absolute value, producing a
    // complex-valued input to the DFT part.
    //
    // Precondition: workspace_bp_ is assumed to be filled with the
    // output of the forward-prop FFT (i.e., the input to the implicit
    // "Abs" calculation).
    internal::ApplyAbsGradientUpdate(local_grad_wrt_output, workspace_);

    // Next we apply the gradient of the DFT, which is... just the
    // DFT. Aren't linear operators fun??
    fft_impl_.compute_forward(workspace_);

    // Finally, we need to transform the output of the DFT, which is
    // complex valued, back to the real world.
    internal::MyRealPart(workspace_, local_grad_wrt_input);
  }

  void setup_fp(RealMatType const& input)
  {
    if (input.Height() != get_linear_size(full_dims_))
      LBANN_ERROR("Invalid input size.");

    auto const num_samples = input.Width();
    auto const output_height = get_linear_size(full_dims_);

    // Nothing to do; short-circuit.
    workspace_.Resize(output_height, num_samples);
    fft_impl_.setup_forward(workspace_, full_dims_);
    fft_impl_.setup_backward(workspace_, full_dims_);
  }

  void setup_bp(RealMatType const& grad_wrt_output) const
  {
    // Nothing to do here. The FFTW plan will be the same for both FP
    // and BP, and the workspace may be holding important state. For
    // debugging purposes, let's just verify the sizes.
    auto const expected_height = get_linear_size(full_dims_);
    if (grad_wrt_output.Height() != expected_height)
      LBANN_ERROR("Gradient with respect to output has "
                  "unexpected height (",
                  grad_wrt_output.Height(),
                  "). Expected height=",
                  expected_height,
                  ".");
  }

private:
  FFTBackend<El::Complex<T>, D> fft_impl_;
  /** @brief Cache the output of the fp DFT. */
  mutable El::Matrix<El::Complex<T>, D> workspace_;
  std::vector<int> full_dims_;

}; // struct dft_abs_impl

// Public
template <typename T, El::Device D>
dft_abs_layer<T, D>::dft_abs_layer(lbann_comm* const comm)
  : data_type_layer<T>(comm)
{}

template <typename T, El::Device D>
dft_abs_layer<T, D>::~dft_abs_layer()
{}

// Protected
template <typename T, El::Device D>
void dft_abs_layer<T, D>::setup_dims()
{
  data_type_layer<T>::setup_dims();
  this->set_output_dims(this->get_input_dims());
  pimpl_ = std::make_unique<dft_abs_impl<T, D>>(this->get_input_dims());
}

template <typename T, El::Device D>
void dft_abs_layer<T, D>::fp_compute()
{
  using LocalMatT = El::Matrix<T, D>;
  pimpl_->setup_fp(
    static_cast<LocalMatT const&>(this->get_local_prev_activations()));
  pimpl_->do_fp_compute(
    static_cast<LocalMatT const&>(this->get_local_prev_activations()),
    static_cast<LocalMatT&>(this->get_local_activations()));
}

template <typename T, El::Device D>
void dft_abs_layer<T, D>::bp_compute()
{
  using LocalMatT = El::Matrix<T, D>;
  pimpl_->setup_bp(
    static_cast<LocalMatT const&>(this->get_local_prev_error_signals()));
  pimpl_->do_bp_compute(
    static_cast<LocalMatT const&>(this->get_local_prev_error_signals()),
    static_cast<LocalMatT&>(this->get_local_error_signals()));
}

template <typename T, El::Device D>
dft_abs_layer<T, D>::dft_abs_layer(dft_abs_layer const& other)
  : data_type_layer<T>(other),
    pimpl_(other.pimpl_ ? std::make_unique<dft_abs_impl<T, D>>(*(other.pimpl_))
                        : nullptr)
{}

#ifdef LBANN_HAS_FFTW_FLOAT
template class dft_abs_layer<float, El::Device::CPU>;
#endif // LBANN_HAS_FFTW_FLOAT
#if defined(LBANN_HAS_DOUBLE) && defined(LBANN_HAS_FFTW_DOUBLE)
template class dft_abs_layer<double, El::Device::CPU>;
#endif // defined(LBANN_HAS_DOUBLE) && defined (LBANN_HAS_FFTW_DOUBLE)

#ifdef LBANN_HAS_GPU
template class dft_abs_layer<float, El::Device::GPU>;
#ifdef LBANN_HAS_DOUBLE
template class dft_abs_layer<double, El::Device::GPU>;
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_GPU

} // namespace lbann
