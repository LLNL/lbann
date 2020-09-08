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
#ifndef LBANN_UTILS_CUFFT_WRAPPER_HPP_
#define LBANN_UTILS_CUFFT_WRAPPER_HPP_

#include <lbann/base.hpp>
#include <lbann/utils/dim_helpers.hpp>
#include <lbann/utils/exception.hpp>
#include <lbann/utils/fft_common.hpp>

#include <cufft.h>

#define LBANN_CHECK_CUFFT(cmd)                                          \
  do {                                                                  \
    auto const lbann_check_cufft_result_ = (cmd);                       \
    if (lbann_check_cufft_result_ != CUFFT_SUCCESS) {                   \
      LBANN_ERROR(                                                      \
        "cuFFT error!\n\n"                                              \
        "      cmd: " #cmd "\n"                                         \
        "   result: ",                                                  \
        lbann::cufft::value_as_string(lbann_check_cufft_result_), "\n"  \
        "  message: ",                                                  \
        lbann::cufft::result_string(lbann_check_cufft_result_),"\n\n"); \
    }                                                                   \
  } while (0)

namespace lbann
{
namespace cufft
{

/** @brief The stringified name of the enumerated value. */
std::string value_as_string(cufftResult_t);
/** @brief The docstring for the given result. */
std::string result_string(cufftResult_t);

template <typename T>
struct cuFFTTypeT;

template <> struct cuFFTTypeT<float> { using type = float; };
template <> struct cuFFTTypeT<double> { using type = double; };
template <>
struct cuFFTTypeT<El::Complex<float>> { using type = cufftComplex; };
template <>
struct cuFFTTypeT<El::Complex<double>> { using type = cufftDoubleComplex; };

template <typename T>
using cuFFTType = typename cuFFTTypeT<T>::type;

template <typename T>
auto AsCUFFTType(T* buffer)
{
  return reinterpret_cast<cuFFTType<T>*>(buffer);
}

/** @brief Alias around the C-compatible API */
template <typename InType, typename OutType>
struct cuFFTExecutor;

template <>
struct cuFFTExecutor<El::Complex<float>, El::Complex<float>>
{
  static constexpr auto transform_type = CUFFT_C2C;
  static void Execute(cufftHandle plan,
                      El::Complex<float>* input_data,
                      El::Complex<float>* output_data,
                      int direction)
  {
    LBANN_CHECK_CUFFT(
      cufftExecC2C(plan,
                   AsCUFFTType(input_data),
                   AsCUFFTType(output_data),
                   direction));
  }
};// struct cuFFTExecutor<Complex<float>, Complex<float>>

template <>
struct cuFFTExecutor<El::Complex<double>, El::Complex<double>>
{
  static constexpr auto transform_type = CUFFT_Z2Z;
  static void Execute(cufftHandle plan,
                      El::Complex<double>* input_data,
                      El::Complex<double>* output_data,
                      int direction)
  {
    LBANN_CHECK_CUFFT(
      cufftExecZ2Z(plan,
                   AsCUFFTType(input_data),
                   AsCUFFTType(output_data),
                   direction));
  }
};// struct cuFFTExecutor<Complex<double>, Complex<double>>

/** @brief Wrapper around cuFFT
 *
 *  The main constraint is that the sample data to which the DFT will
 *  be applied must be fully packed. Batches do not need to be fully
 *  packed, but all lower dimensions do. For example, to compute the
 *  DFT of each feature map in a batch of N samples with C feature
 *  maps of size HxW per sample, the input matrix must have width N
 *  and each column must be CHW-packed, in the cuDNN sense.
 *
 *  This class is structured to match the FFTWWrapper, even though the
 *  cuFFT interface is simpler and better in many ways.
 */
template <typename InputTypeT>
class cuFFTWrapper
{
public:
  using InputType = InputTypeT;
  using OutputType = ToComplex<InputType>;

  using RealType = ToReal<InputType>;
  using ComplexType = ToComplex<InputType>;

  using RealMatType = El::Matrix<RealType, El::Device::GPU>;
  using ComplexMatType = El::Matrix<ComplexType, El::Device::GPU>;

  using ComplexBufferType = El::simple_buffer<ComplexType, El::Device::GPU>;

  using InputMatType = El::Matrix<InputType, El::Device::GPU>;
  using OutputMatType = El::Matrix<OutputType, El::Device::GPU>;

  using ExecutorType = cuFFTExecutor<InputType, OutputType>;
  using PlanType = cufftHandle;

private:
  struct InternalPlanType
  {
    size_t worksize_ = 0ULL;
    PlanType plan_ = 0;
    int num_samples_ = -1; // It's just an int in the basic interface
    InternalPlanType(PlanType plan, size_t worksize, int n)
      : worksize_{worksize},
        plan_{plan},
        num_samples_{n}
    {}
    ~InternalPlanType()
    {
      if (plan_ != 0)
      {
        cufftDestroy(plan_);
        plan_ = 0;
      }
    }
    InternalPlanType(InternalPlanType&& other) noexcept
      : worksize_{other.worksize_},
        plan_{other.plan_},
        num_samples_{other.num_samples_}
    {
      other.worksize_ = 0ULL;
      other.plan_ = 0;
      other.num_samples_ = -1;
    }
  };// struct InternalPlanType

public:
  cuFFTWrapper() = default;
  ~cuFFTWrapper() = default;
  // Movable, not copyable.
  cuFFTWrapper(cuFFTWrapper&& other) noexcept = default;
  cuFFTWrapper(cuFFTWrapper const&) = delete;
  /** @brief Setup the forward transform.
   *  @param in Input array; must be allocated, could be overwritten.
   *  @param out Output array; must be allocated, could be overwritten.
   *  @param full_dims Fold dimensions for the tensors in columns of
   *                   in/out. The format is expected to be
   *                   [num_feature_maps, feature_map_dims].
   */
  void setup_forward(InputMatType& in,
                     OutputMatType& out,
                     std::vector<int> const& full_dims)
  {
    setup_common(in, out, full_dims);
  }
  /** @brief Setup an in-place forward transform.
   *  @param in Input array; must be allocated, could be overwritten.
   *  @param full_dims Fold dimensions for the tensors in columns of
   *                   in/out. The format is expected to be
   *                   [num_feature_maps, feature_map_dims].
   */
  void setup_forward(InputMatType& in,
                     std::vector<int> const& full_dims)
  {
    /// @todo Assert this is ok for R2C cases!!!
    return setup_forward(in, in, full_dims);
  }

  /** @brief Setup the backward (inverse) transform.
   *  @param in Input array; must be allocated, could be overwritten.
   *  @param out Output array; must be allocated, could be overwritten.
   *  @param full_dims Fold dimensions for the tensors in columns of
   *                   in/out. The format is expected to be
   *                   [num_feature_maps, feature_map_dims].
   */
  void setup_backward(OutputMatType& in,
                      InputMatType& out,
                      std::vector<int> const& full_dims)
  {
    return setup_common(in, out, full_dims);
  }

  /** @brief Setup the in-place backward (inverse) transform.
   *  @param in Input array; must be allocated, could be overwritten.
   *  @param full_dims Fold dimensions for the tensors in columns of
   *                   in/out. The format is expected to be
   *                   [num_feature_maps, feature_map_dims].
   */
  void setup_backward(OutputMatType& in,
                      std::vector<int> const& full_dims)
  {
    return setup_backward(in, in, full_dims);
  }

  void compute_forward(InputMatType& in, OutputMatType& out) const
  {
    return compute_common(in, out, CUFFT_FORWARD);
  }

  void compute_forward(InputMatType& in) const
  {
    return compute_common(in, in, CUFFT_FORWARD);
  }

  void compute_backward(OutputMatType& in, InputMatType& out) const
  {
    return compute_common(in, out, CUFFT_INVERSE);
  }

  void compute_backward(OutputMatType& in) const
  {
    return compute_common(in, in, CUFFT_INVERSE);
  }

private:
  void compute_common(OutputMatType& in, InputMatType& out, int dir) const
  {
    auto const num_samples = in.Width();
    if (num_samples == 0)
      return;

    auto const good_plan =
      std::find_if(cbegin(plans_), cend(plans_),
                   [num_samples](InternalPlanType const& a) {
                     return a.num_samples_ == num_samples;
                   });
    if (good_plan == cend(plans_))
      LBANN_ERROR("No valid FFTW plan found.");

    // Setup the workspace
    ComplexBufferType workspace(good_plan->worksize_,
                                El::SyncInfoFromMatrix(out));
    LBANN_CHECK_CUFFT(
      cufftSetWorkArea(good_plan->plan_, workspace.data()));

    // Run the FFT
    bool const contiguous_samples =
      (in.Contiguous()) && (out.Contiguous());
    if (contiguous_samples)
    {
      ExecutorType::Execute(good_plan->plan_, in.Buffer(), out.Buffer(), dir);
    }
    else
    {
      auto num_batches = in.Width();
      for (El::Int ii = 0; ii < num_batches; ++ii)
      {
        ExecutorType::Execute(good_plan->plan_,
                              in.Buffer() + ii*in.LDim(),
                              out.Buffer() + ii*out.LDim(),
                              dir);
      }
    }
  }

  template <typename InMatT, typename OutMatT>
  void setup_common(InMatT& in,
                    OutMatT& out,
                    std::vector<int> const& full_dims)
  {
    using in_data_type = typename InMatT::value_type;
    using out_data_type = typename OutMatT::value_type;
    using Dims = fft::DimsHelper<in_data_type, out_data_type>;

    // Look for an acceptable plan
    int const num_samples = in.Width();
    if (num_samples == 0)
      return;

    auto const good_plan =
      std::find_if(cbegin(plans_), cend(plans_),
                   [num_samples](InternalPlanType const& a) {
                     return a.num_samples_ == num_samples;
                   });

    // We don't have a plan for this yet; let's create one!
    if (good_plan == cend(plans_))
    {
      PlanType plan;
      size_t workspace_size = 0ULL;

      // This is annoying... I could const_cast, but I'm not 100%
      // certain cuFFT doesn't change this data.
      std::vector<int> full_dims_mutable(full_dims);
      LBANN_CHECK_CUFFT(cufftCreate(&plan));
      LBANN_CHECK_CUFFT(
        cufftSetStream(plan, SyncInfoFromMatrix(out).Stream()));
      // We'll handle our own workspace
      LBANN_CHECK_CUFFT(cufftSetAutoAllocation(plan, 0));

      auto const& input_dims = Dims::input_dims(full_dims);
      auto const& output_dims = Dims::output_dims(full_dims);
      int const num_feature_maps = full_dims.front();
      int const feature_map_ndims = full_dims.size()-1;
      bool const contiguous_samples =
        (in.Contiguous()) && (out.Contiguous());

      if (feature_map_ndims > 3 || feature_map_ndims == 0)
        LBANN_ERROR("Only 1-, 2-, and 3-D FFTs are supported in cuFFT.");

      int const input_feature_map_size
        = get_linear_size(feature_map_ndims, input_dims.data() + 1);
      int const output_feature_map_size
        = get_linear_size(feature_map_ndims, output_dims.data() + 1);

      // Handle the easy case. In this case, all the FFTs to be done
      // are contiguous in memory. Super! Let's just set it up to do
      // them all as one big batch.
      if (contiguous_samples)
      {
        int const num_transforms = num_samples*num_feature_maps;
        LBANN_CHECK_CUFFT(
          cufftMakePlanMany(
            plan, feature_map_ndims, full_dims_mutable.data()+1,
            nullptr, 1, input_feature_map_size,
            nullptr, 1, output_feature_map_size,
            ExecutorType::transform_type,
            num_transforms,
            &workspace_size));
      }
      else
      {
        // In this case, we apply the FFT to each sample, and, come
        // execution time, we will loop over the samples. (An
        // alternative might pick whether to loop over samples or
        // channels, whichever is fewer in number. However, this is a
        // book-keeping headache.)
        int const num_transforms = num_feature_maps;
        LBANN_CHECK_CUFFT(
          cufftMakePlanMany(
            plan, feature_map_ndims, full_dims_mutable.data()+1,
            nullptr, 1, input_feature_map_size,
            nullptr, 1, output_feature_map_size,
            ExecutorType::transform_type,
            num_transforms,
            &workspace_size));
      }

      if (plan == 0)
        LBANN_ERROR("cuFFT plan construction failed "
                    "but cuFFT reported no errors.");

      plans_.emplace_back(plan, workspace_size, num_samples);
    }
  }

private:
  // These are likely to be so few in number that a linear search is
  // going to be fine.
  std::vector<InternalPlanType> plans_;

};// class cuFFTWrapper

}// namespace cufft
}// namespace lbann
#endif // LBANN_UTILS_CUFFT_WRAPPER_HPP_
