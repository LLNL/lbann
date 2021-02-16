#ifndef OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_
#define OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_

// Some common includes
#include <lbann_config.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>
#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>


// This header should only be used in the unit testing code, so this
// is fine.
using namespace h2::meta;

// Get the description out as a string. Useful for comparing objects
// that might not expose accessor functions for all metadata.
template <typename ObjectType>
std::string desc_string(ObjectType const& opt)
{
  std::ostringstream desc;
  desc << opt.get_description();
  return desc.str();
}

// Simple groups of Output/Input archive types. The car is the output
// archive, the cadr is the input archive. Accessor metafunctions are
// defined below.

using BinaryArchiveTypes = TL<cereal::BinaryOutputArchive,
                              cereal::BinaryInputArchive>;

using XMLArchiveTypes = TL<cereal::XMLOutputArchive,
                           cereal::XMLInputArchive>;

// A basic "type pack" for things related to the serialization tests.
template <typename OptimizerT,
          typename BuilderT,
          typename ArchiveTypes>
struct TestOptimizer
{
  using OptimizerType = OptimizerT;
  using BuilderType = BuilderT;
  using OutputArchiveType = tlist::Car<ArchiveTypes>;
  using InputArchiveType = tlist::Cadr<ArchiveTypes>;
};

// These are accessor metafunctions to retreive information from
// the type packs used to construct the test cases.
template <typename TypePack>
using GetOptimizerType = typename TypePack::OptimizerType;

template <typename TypePack>
using GetBuilderType = typename TypePack::BuilderType;

template <typename TypePack>
using GetOutputArchiveType = typename TypePack::OutputArchiveType;

template <typename TypePack>
using GetInputArchiveType = typename TypePack::InputArchiveType;

// We need to determine the combinations of types and
// archives that we're going to use for these tests. This needs to
// happen here because embedding preprocessor macros in the arguments
// to preprocessor macros (#ifdef/#endif blocks, e.g.) is not allowed.
//
// Basically, we need the full tensor product of:
//
//   (float,double,[cpu_fp16,[fp16]]) x (Binary,XML)Archives
//
// The test case, then, will be quite simple: construct a stateful
// optimizer, serialize it to the archive, deserialize it to a new
// archive, and then compare the optimizer metadata.

#define CORE_TEMPLATE_ARG_LIST    \
  (float, BinaryArchiveTypes),    \
  (float, XMLArchiveTypes),       \
  (double, BinaryArchiveTypes),   \
  (double, XMLArchiveTypes)

#ifdef LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU_FP16
#define TEMPLATE_ARG_LIST                  \
  ( CORE_TEMPLATE_ARG_LIST,                \
    (lbann::cpu_fp16, BinaryArchiveTypes), \
    (lbann::cpu_fp16, XMLArchiveTypes),    \
    (lbann::fp16, BinaryArchiveTypes),     \
    (lbann::fp16, XMLArchiveTypes) )
#else
#define TEMPLATE_ARG_LIST                  \
  ( CORE_TEMPLATE_ARG_LIST,                \
    (lbann::cpu_fp16, BinaryArchiveTypes), \
    (lbann::cpu_fp16, XMLArchiveTypes) )
#endif // LBANN_HAS_GPU_FP16
#else
#define TEMPLATE_ARG_LIST         \
  ( CORE_TEMPLATE_ARG_LIST )
#endif // LBANN_HAS_HALF

#endif // OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_
