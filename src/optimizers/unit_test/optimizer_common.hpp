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

using Fp16Types = TL<lbann::cpu_fp16
#ifdef LBANN_HAS_GPU_FP16
                     , lbann::fp16
#endif // LBANN_HAS_GPU_FP16
                     >;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
template <typename T>
using BinaryArchiveTypeBundle = TL<T,
                                   cereal::BinaryOutputArchive,
                                   cereal::BinaryInputArchive>;
using BinaryArchiveTypes = tlist::ExpandTL<BinaryArchiveTypeBundle, Fp16Types>;
#else
using BinaryArchiveTypes = tlist::Empty;
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
template <typename T>
using XMLArchiveTypeBundle = TL<T,
                                cereal::XMLOutputArchive,
                                cereal::XMLInputArchive>;
using XMLArchiveTypes = tlist::ExpandTL<XMLArchiveTypeBundle, Fp16Types>;
#else
using XMLArchiveTypes = tlist::Empty;
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

using AllArchiveTypes = tlist::Append<BinaryArchiveTypes,
                                      XMLArchiveTypes>;

#endif // OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_
