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
#ifndef LBANN_UTILS_FFTW_WRAPPER_HPP_
#define LBANN_UTILS_FFTW_WRAPPER_HPP_

#include <lbann/base.hpp>
#include <lbann/utils/dim_helpers.hpp>
#include <lbann/utils/exception.hpp>
#include <lbann/utils/fft_common.hpp>

#include <fftw3.h>

namespace lbann
{

namespace fftw
{

template <typename T>
struct FFTWTypeT;

template <> struct FFTWTypeT<float> { using type = float; };
template <> struct FFTWTypeT<double> { using type = double; };
template <>
struct FFTWTypeT<El::Complex<float>> { using type = fftwf_complex; };
template <>
struct FFTWTypeT<El::Complex<double>> { using type = fftw_complex; };

template <typename T>
using FFTWType = typename FFTWTypeT<T>::type;

template <typename T>
auto AsFFTWType(T* buffer)
{
  return reinterpret_cast<FFTWType<T>*>(buffer);
}

template <typename InputT, typename OutputT>
struct FFTWTraits;

#define BUILD_FFTW_R2C_TRAITS(INTYPE, FFTW_PREFIX)                      \
  template <>                                                           \
  struct FFTWTraits<INTYPE, El::Complex<INTYPE>>                        \
  {                                                                     \
    using plan_type = FFTW_PREFIX ## _plan;                             \
    using iodim_type = FFTW_PREFIX ## _iodim;                           \
    static constexpr auto plan_many_fwd = &FFTW_PREFIX ## _plan_many_dft_r2c; \
    static constexpr auto plan_many_bwd = &FFTW_PREFIX ## _plan_many_dft_c2r; \
    static constexpr auto plan_guru_fwd = &FFTW_PREFIX ## _plan_guru_dft_r2c; \
    static constexpr auto plan_guru_bwd = &FFTW_PREFIX ## _plan_guru_dft_c2r; \
    static constexpr auto execute_plan_fwd = &FFTW_PREFIX ## _execute_dft_r2c; \
    static constexpr auto execute_plan_bwd = &FFTW_PREFIX ## _execute_dft_c2r; \
    static constexpr auto destroy_plan = &FFTW_PREFIX ## _destroy_plan; \
    static constexpr auto plain_execute = &FFTW_PREFIX ## _execute; \
  }

#define BUILD_FFTW_C2C_TRAITS(INTYPE, FFTW_PREFIX)                      \
  template <>                                                           \
  struct FFTWTraits<El::Complex<INTYPE>, El::Complex<INTYPE>>           \
  {                                                                     \
    using plan_type = FFTW_PREFIX ## _plan;                             \
    using iodim_type = FFTW_PREFIX ## _iodim;                           \
    static constexpr auto execute_plan_fwd = &FFTW_PREFIX ## _execute_dft; \
    static constexpr auto execute_plan_bwd = &FFTW_PREFIX ## _execute_dft; \
    static constexpr auto destroy_plan = &FFTW_PREFIX ## _destroy_plan; \
    static constexpr auto plain_execute = &FFTW_PREFIX ## _execute;     \
    static plan_type plan_many_fwd(                                     \
      int rank, const int *n, int howmany,                              \
      FFTW_PREFIX ## _complex *in, const int *inembed, int istride, int idist, \
      FFTW_PREFIX ## _complex *out, const int *onembed, int ostride, int odist, \
      unsigned flags) {                                                 \
      return FFTW_PREFIX ## _plan_many_dft(                             \
        rank, n, howmany, in, inembed, istride, idist,                   \
        out, onembed, ostride, odist, FFTW_FORWARD, flags);             \
    }                                                                   \
    static plan_type plan_many_bwd(                                     \
      int rank, const int *n, int howmany,                              \
      FFTW_PREFIX ## _complex *in, const int *inembed, int istride, int idist, \
      FFTW_PREFIX ## _complex *out, const int *onembed, int ostride, int odist, \
      unsigned flags) {                                                 \
      return FFTW_PREFIX ## _plan_many_dft(                             \
        rank, n, howmany, in, inembed, istride, idist,                  \
        out, onembed, ostride, odist, FFTW_BACKWARD, flags);            \
    }                                                                   \
    static plan_type plan_guru_fwd(                                     \
      int rank, const FFTW_PREFIX ## _iodim *dims,                      \
      int howmany_rank, const FFTW_PREFIX ## _iodim *howmany_dims,      \
      FFTW_PREFIX ## _complex *in, FFTW_PREFIX ## _complex *out,        \
      unsigned flags) {                                                 \
      return FFTW_PREFIX ## _plan_guru_dft(                             \
        rank, dims, howmany_rank, howmany_dims,                         \
        in, out, FFTW_FORWARD, flags);                                  \
    }                                                                   \
    static plan_type plan_guru_bwd(                                     \
      int rank, const FFTW_PREFIX ## _iodim *dims,                      \
      int howmany_rank, const FFTW_PREFIX ## _iodim *howmany_dims,      \
      FFTW_PREFIX ## _complex *in, FFTW_PREFIX ## _complex *out,        \
      unsigned flags) {                                                 \
      return FFTW_PREFIX ## _plan_guru_dft(                             \
        rank, dims, howmany_rank, howmany_dims,                         \
        in, out, FFTW_BACKWARD, flags);                                 \
    }                                                                   \
  }

BUILD_FFTW_R2C_TRAITS(float, fftwf);
BUILD_FFTW_R2C_TRAITS(double, fftw);

BUILD_FFTW_C2C_TRAITS(float, fftwf);
BUILD_FFTW_C2C_TRAITS(double, fftw);

/** @brief Wrapper around FFTW
 *
 *  The main constraint is that the sample data to which the DFT will
 *  be applied must be fully packed. Batches do not need to be fully
 *  packed, but all lower dimensions do. For example, to compute the
 *  DFT of each feature map in a batch of N samples with C feature
 *  maps of size HxW per sample, the input matrix must have width N
 *  and each column must be CHW-packed, in the cuDNN sense.
 */
template <typename InputTypeT>
class FFTWWrapper
{
public:
  using InputType = InputTypeT;
  using OutputType = ToComplex<InputType>;

  using RealType = ToReal<InputType>;
  using ComplexType = ToComplex<InputType>;
  using TraitsType = FFTWTraits<InputType, ToComplex<InputType>>;

  using RealMatType = El::Matrix<RealType, El::Device::CPU>;
  using ComplexMatType = El::Matrix<ComplexType, El::Device::CPU>;

  using InputMatType = El::Matrix<InputType, El::Device::CPU>;
  using OutputMatType = El::Matrix<OutputType, El::Device::CPU>;

  using PlanType = typename TraitsType::plan_type;

private:
  struct InternalPlanType
  {
    PlanType plan_ = nullptr;
    int num_samples_ = -1; // It's just an int in fftw
    InternalPlanType(PlanType plan, int n)
      : plan_{plan},
        num_samples_{n}
    {}
    ~InternalPlanType()
    {
      if (plan_ != nullptr)
      {
        TraitsType::destroy_plan(plan_);
        plan_ = nullptr;
      }
    }
    InternalPlanType(InternalPlanType&& other) noexcept
      : plan_{other.plan_},
        num_samples_{other.num_samples_}
    {
      other.plan_ = nullptr;
      other.num_samples_ = -1;
    }
  };// struct InternalPlanType

public:
  FFTWWrapper() = default;
  ~FFTWWrapper() = default;
  // Movable, not copyable.
  FFTWWrapper(FFTWWrapper&& other) noexcept = default;
  FFTWWrapper(FFTWWrapper const&) = delete;
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
    setup_common(in, out, full_dims, fwd_plans_,
                 TraitsType::plan_many_fwd,
                 TraitsType::plan_guru_fwd);
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
    setup_forward(in, in, full_dims);
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
    setup_common(in, out, full_dims, bwd_plans_,
                 TraitsType::plan_many_bwd,
                 TraitsType::plan_guru_bwd);
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
    setup_backward(in, in, full_dims);
  }

  void compute_forward(InputMatType& in, OutputMatType& out) const
  {
    auto const num_samples = in.Width();
    auto const good_plan =
      std::find_if(cbegin(fwd_plans_), cend(fwd_plans_),
                   [num_samples](InternalPlanType const& a) {
                     return a.num_samples_ == num_samples;
                   });
    if (good_plan == cend(fwd_plans_))
      LBANN_ERROR("No valid FFTW plan found.");

    // Initial tests suggest there's no performance reason to *not*
    // use the "new-array" interface.
    TraitsType::execute_plan_fwd(
      good_plan->plan_,
      AsFFTWType(in.Buffer()),
      AsFFTWType(out.Buffer()));
  }

  void compute_forward(InputMatType& in) const
  {
    return compute_forward(in, in);
  }

  void compute_backward(OutputMatType& in, InputMatType& out) const
  {
    auto const num_samples = in.Width();
    auto const good_plan =
      std::find_if(cbegin(bwd_plans_), cend(bwd_plans_),
                   [num_samples](InternalPlanType const& a) {
                     return a.num_samples_ == num_samples;
                   });
    if (good_plan == cend(bwd_plans_))
      LBANN_ERROR("No valid FFTW plan found.");

    // Initial tests suggest there's no performance reason to *not*
    // use the "new-array" interface.
    TraitsType::execute_plan_bwd(
      good_plan->plan_,
      AsFFTWType(in.Buffer()),
      AsFFTWType(out.Buffer()));
  }

  void compute_backward(OutputMatType& in) const
  {
    return compute_backward(in, in);
  }

private:

  template <typename InMatT, typename OutMatT,
            typename SetupManyFunctorT, typename SetupGuruFunctorT>
  void setup_common(InMatT& in,
                    OutMatT& out,
                    std::vector<int> const& full_dims,
                    std::vector<InternalPlanType>& plans,
                    SetupManyFunctorT many_functor,
                    SetupGuruFunctorT guru_functor)
  {
    using in_data_type = typename InMatT::value_type;
    using out_data_type = typename OutMatT::value_type;
    using Dims = fft::DimsHelper<in_data_type, out_data_type>;

    // Look for an acceptable plan
    int const num_samples = in.Width();
    auto const good_plan =
      std::find_if(cbegin(plans), cend(plans),
                   [num_samples](InternalPlanType const& a) {
                     return a.num_samples_ == num_samples;
                   });

    // We don't have a plan for this yet; let's create one!
    if (good_plan == cend(plans))
    {
      PlanType plan;

      auto const& input_dims = Dims::input_dims(full_dims);
      auto const& output_dims = Dims::output_dims(full_dims);
      int const num_feature_maps = full_dims.front();
      int const feature_map_ndims = full_dims.size()-1;
      bool const contiguous_samples =
        (in.Contiguous()) && (out.Contiguous());

      // Handle the easy case
      if (contiguous_samples)
      {
        int const num_transforms = num_samples*num_feature_maps;
        int const input_feature_map_size
          = get_linear_size(feature_map_ndims, input_dims.data() + 1);
        int const output_feature_map_size
          = get_linear_size(feature_map_ndims, output_dims.data() + 1);
        plan = many_functor(
          feature_map_ndims, full_dims.data()+1, num_transforms,
          AsFFTWType(in.Buffer()), nullptr, 1, input_feature_map_size,
          AsFFTWType(out.Buffer()), nullptr, 1, output_feature_map_size,
          /*flags=*/0);//FFTW_PRESERVE_INPUT);
      }
      else
      {
        using IODimType = typename TraitsType::iodim_type;

        std::vector<IODimType> dims(feature_map_ndims), how_many(2);

        auto input_strides = get_packed_strides(input_dims);
        auto output_strides = get_packed_strides(output_dims);

        // Setup the "dims"
        for (int d = 0; d < feature_map_ndims; ++d)
        {
          dims[d].n = full_dims[d+1];
          dims[d].is = input_strides[d+1];
          dims[d].os = output_strides[d+1];
        }

        // Setup the "howmany"
        how_many[0].n = num_feature_maps;
        how_many[0].is = input_strides.front();
        how_many[0].os = output_strides.front();

        how_many[1].n = num_samples;
        how_many[1].is = in.LDim();
        how_many[1].os = out.LDim();

        plan = guru_functor(
          dims.size(), dims.data(),
          how_many.size(), how_many.data(),
          AsFFTWType(in.Buffer()), AsFFTWType(out.Buffer()),
          /*flags=*/0);//FFTW_PRESERVE_INPUT);
      }

      if (plan == nullptr)
        LBANN_ERROR(__PRETTY_FUNCTION__,
                    ": FFTW plan construction failed.\n"
                    "  contiguous: ", contiguous_samples);

      plans.emplace_back(plan, num_samples);
    }
  }

private:
  // These are likely to be so few in number that a linear search is
  // going to be fine.
  std::vector<InternalPlanType> fwd_plans_;
  std::vector<InternalPlanType> bwd_plans_;

};// class FFTWWrapper

}// namespace fftw

}// namespace lbann
#endif // LBANN_UTILS_FFTW_WRAPPER_HPP_
