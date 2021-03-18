#include <catch2/catch.hpp>
#include <lbann/base.hpp>

#include <lbann/utils/random.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#include "MPITestHelpers.hpp"

/** @brief Statistic for Anderson-Darling normality test
 *
 *  A higher statistic implies greater confidence in non-normality.
 *
 *  See:
 *
 *  George Marsaglia and John Marsaglia. "Evaluating the
 *  Anderson-Darling distribution." Journal of statistical software 9,
 *  no. 2 (2004): 1-5.
 *
 *  @todo Implement as uniformity test and apply to @c uniform_fill.
 */
double anderson_darling_test(
  std::vector<double>& values,
  double mean,
  double stddev)
{

  // Convert values to CDF form
  std::sort(values.begin(), values.end());
  std::for_each(
    values.begin(),
    values.end(),
    [mean, stddev] (double& x)
    {
      x = (x-mean) / stddev;              // z-score
      x = std::erfc(-x/std::sqrt(2)) / 2; // Normal CDF
    });

  // Perform Anderson-Darling test
  const int N = values.size();
  double sum = 0;
  for (int i=0; i<N; ++i)
  {
    sum += (2*i+1) * ( std::log(values[i]) + std::log(1-values[N-1-i]) );
  }
  return -N - sum/N;

}

/** @brief Critical value for Anderson-Darling normality test
 *
 *  A statistic of 1.9329578327415937304 corresponds to a significance
 *  of 0.1, 2.4923671600494096176 to 0.05, and 3.8781250216053948842
 *  to 0.01.
 */
constexpr double anderson_darling_critical_value = 2.4923671600494096176;

TEST_CASE("Anderson-Darling normality test", "[random][utilities]")
{

  // Parameters
  const double mean = 12.25;
  const double stddev = 07.04;

  // Applied inverse of normal CDF to uniform range in [0,1)
  const std::vector<double> normal_z_scores = {
    -2.24140273, -1.78046434, -1.53412054, -1.35631175, -1.21333962,
    -1.09162037, -0.98423496, -0.88714656, -0.79777685, -0.71436744,
    -0.63565701, -0.56070303, -0.48877641, -0.41929575, -0.35178434,
    -0.28584087, -0.22111871, -0.15731068, -0.09413741, -0.03133798,
    0.03133798,  0.09413741,  0.15731068,  0.22111871,  0.28584087,
    0.35178434,  0.41929575,  0.48877641,  0.56070303,  0.63565701,
    0.71436744,  0.79777685,  0.88714656,  0.98423496,  1.09162037,
    1.21333962,  1.35631175,  1.53412054,  1.78046434,  2.24140273};

  SECTION("Normal distribution")
  {
    std::vector<double> values = normal_z_scores;
    std::for_each(
      values.begin(),
      values.end(),
      [mean, stddev] (double& x) { x = x*stddev + mean; });
    REQUIRE(
      anderson_darling_test(values, mean, stddev)
      < anderson_darling_critical_value);
  }

  SECTION("Shifted normal distribution")
  {
    std::vector<double> values = normal_z_scores;
    std::for_each(
      values.begin(),
      values.end(),
      [mean, stddev] (double& x) { x = x*stddev + mean; });
    REQUIRE_FALSE(
      anderson_darling_test(values, mean+3, stddev)
      < anderson_darling_critical_value);
  }

  SECTION("Uniform distribution")
  {
    const int N = 654;
    std::vector<double> values;
    for (int i=0; i<N; ++i)
    {
      double x = (i+0.5)/N;       // Uniform in [0,1)
      x = 2*std::sqrt(3)*(x-0.5); // z-score
      values.push_back(x*stddev + mean);
    }
    REQUIRE_FALSE(
      anderson_darling_test(values, mean, stddev)
      < anderson_darling_critical_value);
  }

}

// Enumerate all DistMatrix types. Start by getting all the
// distributions.
template <typename T, El::Device D>
using DistMatrixTypesWithDevice = h2::meta::TL<
  // There is currently a known bug where copying
  // (CIRC,CIRC,CPU)->(CIRC,CIRC,GPU) results in an infinite recursion
  // in Hydrogen. Since we don't actually use this in our code, ignore
  // this case for now.
  //
  // El::DistMatrix<T, El::CIRC, El::CIRC, El::ELEMENT, D>,
  El::DistMatrix<T, El::MC  , El::MR  , El::ELEMENT, D>,
  El::DistMatrix<T, El::MC  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MD  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MR  , El::MC  , El::ELEMENT, D>,
  El::DistMatrix<T, El::MR  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MC  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MD  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MR  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::VC  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::VR  , El::ELEMENT, D>,
  El::DistMatrix<T, El::VC  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::VR  , El::STAR, El::ELEMENT, D>>;

// Enumerate valid combinations of data types and devices
using AllDistMatrixTypes = h2::meta::tlist::Append<
  DistMatrixTypesWithDevice<float, El::Device::CPU>
  , DistMatrixTypesWithDevice<double, El::Device::CPU>
#if defined LBANN_HAS_GPU
  , DistMatrixTypesWithDevice<float, El::Device::GPU>
  , DistMatrixTypesWithDevice<double, El::Device::GPU>
#endif // defined LBANN_HAS_GPU
#if defined LBANN_HAS_CPU_FP16
  , DistMatrixTypesWithDevice<lbann::cpu_fp16, El::Device::CPU>
#endif // defined LBANN_HAS_CPU_FP16
#if defined LBANN_HAS_GPU_FP16
  , DistMatrixTypesWithDevice<lbann::fp16, El::Device::GPU>
#endif // defined LBANN_HAS_GPU_FP16
  >;

// Meta-programming to get data type from matrix
template <typename DistMat>
struct TensorDataTypeStruct;
template <typename T, El::Dist ColDist, El::Dist RowDist, El::DistWrap Wrap, El::Device D>
struct TensorDataTypeStruct<El::DistMatrix<T,ColDist,RowDist,Wrap,D>>
{
  using type = T;
};
template <typename DistMat>
using TensorDataType = typename TensorDataTypeStruct<DistMat>::type;

TEMPLATE_LIST_TEST_CASE(
  "Testing gaussian_fill",
  "[random][utilities][mpi]",
  AllDistMatrixTypes)
{

  // Typedefs
  using DistMatType = TestType;
  using StarMatType = El::DistMatrix<double, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>;

  // Parameters
  const TensorDataType<DistMatType> mean = 12.3f;
  const TensorDataType<DistMatType> stddev = 4.56f;
  const El::Int height = 53;
  const El::Int width = 41;

  // Initialization
  auto& comm = ::unit_test::utilities::current_world_comm();
  lbann::init_random(-1, 0, &comm);

  SECTION("Contiguous matrix")
  {

    // Attempt Anderson-Darling test several times
    bool passed_test = false;
    for (int iter=0; iter<3; ++iter)
    {

      // Fill matrix with random values
      DistMatType mat(comm.get_trainer_grid());
      REQUIRE_NOTHROW(lbann::gaussian_fill(mat, height, width, mean, stddev));

      // Check matrix dimensions
      REQUIRE(mat.Height() == height);
      REQUIRE(mat.Width() == width);

      // Check that values are normally distributed
      StarMatType mat_copy(mat.Grid());
      El::Copy(mat, mat_copy);
      std::vector<double> values;
      for (El::Int col = 0; col < width; ++col)
      {
        for (El::Int row = 0; row < height; ++row)
        {
          values.push_back(mat_copy.Get(row, col));
        }
      }
      if (anderson_darling_test(
            values,
            static_cast<double>(mean),
            static_cast<double>(stddev))
          < anderson_darling_critical_value)
      {
        passed_test = true;
        break;
      }

    }

    // Make sure Anderson-Darling test has succeeded at least once
    REQUIRE(passed_test);

  }

  SECTION("Non-contiguous matrix")
  {

    // Attempt Anderson-Darling test several times
    bool passed_test = false;
    for (int iter=0; iter<3; ++iter)
    {

      // Fill matrix with random values
      DistMatType mat(2*height, width, comm.get_trainer_grid());
      DistMatType mat_view = El::View(mat, El::IR(height/2,height+height/2), El::ALL);
      REQUIRE_NOTHROW(lbann::gaussian_fill(mat_view, height, width, mean, stddev));

      // Check matrix dimensions
      REQUIRE(mat_view.Height() == height);
      REQUIRE(mat_view.Width() == width);
      REQUIRE(mat_view.Viewing());
      if (!mat_view.LockedMatrix().IsEmpty())
      {
        REQUIRE_FALSE(mat_view.Contiguous());
      }

      // Check that values are normally distributed
      StarMatType mat_copy(mat_view.Grid());
      El::Copy(mat_view, mat_copy);
      std::vector<double> values;
      for (El::Int col = 0; col < width; ++col)
      {
        for (El::Int row = 0; row < height; ++row)
        {
          values.push_back(mat_copy.Get(row, col));
        }
      }
      if (anderson_darling_test(
            values,
            static_cast<double>(mean),
            static_cast<double>(stddev))
          < anderson_darling_critical_value)
      {
        passed_test = true;
        break;
      }

    }

    // Make sure Anderson-Darling test has succeeded at least once
    REQUIRE(passed_test);

  }

}
