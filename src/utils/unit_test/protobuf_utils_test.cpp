////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#include "Catch2BasicSupport.hpp"

#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/protobuf/decl.hpp"
#include "lbann/utils/protobuf/impl.hpp"

#include <google/protobuf/descriptor.h>
#include "protobuf_utils_test_messages.pb.h"

#include <string>

static_assert(lbann::protobuf::details::PBCppType<lbann::protobuf::int32> ==
              google::protobuf::FieldDescriptor::CPPTYPE_INT32);
static_assert(lbann::protobuf::details::PBCppType<lbann::protobuf::uint32> ==
              google::protobuf::FieldDescriptor::CPPTYPE_UINT32);
static_assert(lbann::protobuf::details::PBCppType<lbann::protobuf::int64> ==
              google::protobuf::FieldDescriptor::CPPTYPE_INT64);
static_assert(lbann::protobuf::details::PBCppType<lbann::protobuf::uint64> ==
              google::protobuf::FieldDescriptor::CPPTYPE_UINT64);

static_assert(lbann::protobuf::details::PBCppType<double> ==
              google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE);
static_assert(lbann::protobuf::details::PBCppType<float> ==
              google::protobuf::FieldDescriptor::CPPTYPE_FLOAT);

static_assert(lbann::protobuf::details::PBCppType<std::string> ==
              google::protobuf::FieldDescriptor::CPPTYPE_STRING);

static_assert(lbann::protobuf::details::PBCppType<bool> ==
              google::protobuf::FieldDescriptor::CPPTYPE_BOOL);

static_assert(lbann::protobuf::details::PBCppType<lbann_testing::SimpleMesg> ==
              google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE);
static_assert(lbann::protobuf::details::PBCppType<lbann_testing::MyEnum> ==
              google::protobuf::FieldDescriptor::CPPTYPE_ENUM);

#ifdef LBANN_USE_CATCH2_V3
static Catch::Matchers::StringContainsMatcher Contains(std::string const& str)
{
  return Catch::Matchers::ContainsSubstring(str, Catch::CaseSensitive::Yes);
}
#endif // LBANN_USE_CATCH2_V3

TEST_CASE("Basic utilities", "[protobuf][utils]")
{
  lbann_testing::SimpleMesg msg;
  CHECK(lbann::protobuf::message_type(msg) == "SimpleMesg");
}

TEST_CASE("Basic oneof Utilities", "[protobuf][utils]")
{
  std::string const msg_string_str = R"ptext(
my_string: "apples"
another_field: false
)ptext";

  std::string const msg_uint64_str = R"ptext(
my_uint64: 123
another_field: false
)ptext";

  std::string const msg_NO_ONEOF_str = R"ptext(
another_field: false
)ptext";

  lbann_testing::HasAOneofField msg;
  CHECK_NOTHROW(lbann::protobuf::text::fill(msg_string_str, msg));
  CHECK(lbann::protobuf::which_oneof(msg, "my_oneof") == "my_string");

  CHECK_NOTHROW(lbann::protobuf::text::fill(msg_uint64_str, msg));
  CHECK(lbann::protobuf::which_oneof(msg, "my_oneof") == "my_uint64");
  CHECK_THROWS_WITH(
    lbann::protobuf::get_oneof_message(msg, "my_oneof"),
    Contains("Oneof \"my_oneof\" has field \"my_uint64\" set but it is "
             "not of message type."));

  CHECK_NOTHROW(lbann::protobuf::text::fill(msg_NO_ONEOF_str, msg));
  CHECK_THROWS_WITH(
    lbann::protobuf::which_oneof(msg, "my_oneof"),
    Contains("Oneof field \"my_oneof\" in message has not been set."));

  CHECK_THROWS_WITH(
    lbann::protobuf::get_oneof_message(msg, "my_oneof"),
    Contains("Oneof field \"my_oneof\" in message has not been set."));
}

TEST_CASE("Complex oneof manipulation", "[protobuf][utils]")
{
  std::string const msg_msg_str = R"ptext(
my_simple_msg {
  my_string: "Hello world"
}
another_field: false
)ptext";

  lbann_testing::HasAOneofField msg;
  CHECK_NOTHROW(lbann::protobuf::text::fill(msg_msg_str, msg));

  // Verify the "metadata" looks ok.
  CHECK(lbann::protobuf::has_oneof(msg, "my_oneof"));
  CHECK_FALSE(lbann::protobuf::has_oneof(msg, "another_field"));
  CHECK_FALSE(lbann::protobuf::has_oneof(msg, "your_oneof"));
  CHECK(lbann::protobuf::which_oneof(msg, "my_oneof") == "my_simple_msg");

  // Actually get the message out
  lbann_testing::SimpleMesg simple_msg;
  CHECK_NOTHROW(
    simple_msg.CopyFrom(lbann::protobuf::get_oneof_message(msg, "my_oneof")));
  CHECK(simple_msg.my_string() == "Hello world");
}
TEST_CASE("Assign container to repeated protobuf field")
{
  std::vector<lbann::protobuf::uint32> const values{0u,1u,2u,3u};
  lbann_testing::HasRepeatedPODFields msg;
  lbann::protobuf::assign_to_repeated(*msg.mutable_my_uint32s(), values);

  REQUIRE((size_t) msg.my_uint32s_size() == values.size());
  for (size_t ii = 0; ii < values.size() ; ++ii)
    CHECK(msg.my_uint32s(ii) == values[ii]);
}
TEST_CASE("Convert container of streamable objects to space separated string")
{
  std::vector<int> values{0,1,2,3};
  auto my_str = lbann::protobuf::to_space_sep_string(values);

  CHECK(my_str == "0 1 2 3");
}
TEST_CASE("Repeated POD field extraction", "[protobuf][utils]")
{
  std::string const msg_repeated_pod_field = R"ptext(
my_floats: 1.0
my_floats: 2.0
my_floats: 2.0
my_floats: 3.0
my_doubles: 13.0
my_doubles: 13.0
my_doubles: 14.0
my_doubles: 15.0
my_uint32s: 1
my_uint32s: 2
my_uint32s: 3
my_uint32s: 3
my_uint32s: 3
my_int64s: 16
my_int64s: 32
my_int64s: 64
my_enums: ZERO
my_enums: ONE
my_enums: TWO
my_enums: ZERO
my_enums: ONE
my_enums: TWO
)ptext";

  lbann_testing::HasRepeatedPODFields msg;
  CHECK_NOTHROW(lbann::protobuf::text::fill(msg_repeated_pod_field, msg));

  SECTION("Extracting as the wrong type fails")
  {
    CHECK_THROWS_WITH(
      lbann::protobuf::as_vector<lbann::protobuf::uint64>(msg, "my_int64s"),
      Contains("Field has incompatible type"));
  }

  SECTION("As vector - static field")
  {
    CHECK(lbann::protobuf::as_vector(msg.my_floats()) ==
          std::vector<float>{1.f, 2.f, 2.f, 3.f});
    CHECK(lbann::protobuf::as_vector(msg.my_doubles()) ==
          std::vector<double>{13., 13., 14., 15.});
    CHECK(lbann::protobuf::as_vector(msg.my_uint32s()) ==
          std::vector<lbann::protobuf::uint32>{1U, 2U, 3U, 3U, 3U});
    CHECK(lbann::protobuf::as_vector(msg.my_int64s()) ==
          std::vector<lbann::protobuf::int64>{16L, 32L, 64L});
    CHECK(lbann::protobuf::as_vector(msg.my_enums()) ==
          std::vector<lbann::protobuf::int32>{0, 1, 2, 0, 1, 2});
  }

  SECTION("To vector - static field")
  {
    CHECK(lbann::protobuf::to_vector<double>(msg.my_floats()) ==
          std::vector<double>{1., 2., 2., 3.});
    CHECK(lbann::protobuf::to_vector<int>(msg.my_uint32s()) ==
          std::vector<int>{1, 2, 3, 3, 3});
    CHECK(lbann::protobuf::to_vector<int>(msg.my_int64s()) ==
          std::vector<int>{16, 32, 64});
    CHECK(lbann::protobuf::as_vector(msg.my_enums()) ==
          std::vector<lbann::protobuf::int32>{0, 1, 2, 0, 1, 2});
  }

  SECTION("As vector - dynamic field")
  {
    CHECK(lbann::protobuf::as_vector<float>(msg, "my_floats") ==
          std::vector<float>{1.f, 2.f, 2.f, 3.f});
    CHECK(lbann::protobuf::as_vector<double>(msg, "my_doubles") ==
          std::vector<double>{13., 13., 14., 15.});
    CHECK(
      lbann::protobuf::as_vector<lbann::protobuf::uint32>(msg, "my_uint32s") ==
      std::vector<lbann::protobuf::uint32>{1U, 2U, 3U, 3U, 3U});
    CHECK(
      lbann::protobuf::as_vector<lbann::protobuf::int64>(msg, "my_int64s") ==
      std::vector<lbann::protobuf::int64>{16L, 32L, 64L});

    // Enums can be dynamically fetched as int32s or by their enum type.
    CHECK(lbann::protobuf::as_vector<lbann::protobuf::int32>(msg, "my_enums") ==
          std::vector<lbann::protobuf::int32>{0, 1, 2, 0, 1, 2});
    CHECK(lbann::protobuf::as_vector<lbann_testing::MyEnum>(msg, "my_enums") ==
          std::vector<lbann_testing::MyEnum>{lbann_testing::ZERO,
                                             lbann_testing::ONE,
                                             lbann_testing::TWO,
                                             lbann_testing::ZERO,
                                             lbann_testing::ONE,
                                             lbann_testing::TWO});
  }

  SECTION("As set - static field")
  {
    CHECK(lbann::protobuf::as_set(msg.my_floats()) ==
          std::set<float>{1.f, 2.f, 3.f});
    CHECK(lbann::protobuf::as_set(msg.my_doubles()) ==
          std::set<double>{13., 14., 15.});
    CHECK(lbann::protobuf::as_set(msg.my_uint32s()) ==
          std::set<lbann::protobuf::uint32>{1U, 2U, 3U});
    CHECK(lbann::protobuf::as_set(msg.my_int64s()) ==
          std::set<lbann::protobuf::int64>{16L, 32L, 64L});
    CHECK(lbann::protobuf::as_set(msg.my_enums()) ==
          std::set<lbann::protobuf::int32>{0, 1, 2});
  }

  SECTION("To set - static field")
  {
    CHECK(lbann::protobuf::to_set<double>(msg.my_floats()) ==
          std::set<double>{1.f, 2.f, 3.f});
    CHECK(lbann::protobuf::to_set<double>(msg.my_doubles()) ==
          std::set<double>{13., 14., 15.});
    CHECK(lbann::protobuf::to_set<int>(msg.my_uint32s()) ==
          std::set<int>{1, 2, 3});
    CHECK(lbann::protobuf::to_set<int>(msg.my_int64s()) ==
          std::set<int>{16, 32, 64});
    CHECK(lbann::protobuf::to_set<int>(msg.my_enums()) ==
          std::set<int>{0, 1, 2});
  }

  SECTION("As set - dynamic field")
  {
    CHECK(lbann::protobuf::as_set<float>(msg, "my_floats") ==
          std::set<float>{1.f, 2.f, 3.f});
    CHECK(lbann::protobuf::as_set<double>(msg, "my_doubles") ==
          std::set<double>{13., 14., 15.});
    CHECK(lbann::protobuf::as_set<lbann::protobuf::uint32>(msg, "my_uint32s") ==
          std::set<lbann::protobuf::uint32>{1U, 2U, 3U});
    CHECK(lbann::protobuf::as_set<lbann::protobuf::int64>(msg, "my_int64s") ==
          std::set<lbann::protobuf::int64>{16L, 32L, 64L});

    // Enums can be dynamically fetched as int32s or by their enum type.
    CHECK(lbann::protobuf::as_set<lbann::protobuf::int32>(msg, "my_enums") ==
          std::set<lbann::protobuf::int32>{0, 1, 2});
    CHECK(lbann::protobuf::as_set<lbann_testing::MyEnum>(msg, "my_enums") ==
          std::set<lbann_testing::MyEnum>{lbann_testing::ZERO,
                                          lbann_testing::ONE,
                                          lbann_testing::TWO});
  }

  SECTION("As unordered_set - static field")
  {
    CHECK(lbann::protobuf::as_unordered_set(msg.my_floats()) ==
          std::unordered_set<float>{1.f, 2.f, 3.f});
    CHECK(lbann::protobuf::as_unordered_set(msg.my_doubles()) ==
          std::unordered_set<double>{13., 14., 15.});
    CHECK(lbann::protobuf::as_unordered_set(msg.my_uint32s()) ==
          std::unordered_set<lbann::protobuf::uint32>{1U, 2U, 3U});
    CHECK(lbann::protobuf::as_unordered_set(msg.my_int64s()) ==
          std::unordered_set<lbann::protobuf::int64>{16L, 32L, 64L});
    CHECK(lbann::protobuf::as_unordered_set(msg.my_enums()) ==
          std::unordered_set<lbann::protobuf::int32>{0, 1, 2});
  }

  SECTION("To unordered set - static field")
  {
    CHECK(lbann::protobuf::to_unordered_set<double>(msg.my_floats()) ==
          std::unordered_set<double>{1.f, 2.f, 3.f});
    CHECK(lbann::protobuf::to_unordered_set<double>(msg.my_doubles()) ==
          std::unordered_set<double>{13., 14., 15.});
    CHECK(lbann::protobuf::to_unordered_set<int>(msg.my_uint32s()) ==
          std::unordered_set<int>{1, 2, 3});
    CHECK(lbann::protobuf::to_unordered_set<int>(msg.my_int64s()) ==
          std::unordered_set<int>{16, 32, 64});
    CHECK(lbann::protobuf::to_unordered_set<int>(msg.my_enums()) ==
          std::unordered_set<int>{0, 1, 2});
  }

  SECTION("As unordered_set - dynamic field")
  {
    CHECK(lbann::protobuf::as_unordered_set<float>(msg, "my_floats") ==
          std::unordered_set<float>{1.f, 2.f, 3.f});
    CHECK(lbann::protobuf::as_unordered_set<double>(msg, "my_doubles") ==
          std::unordered_set<double>{13., 14., 15.});
    CHECK(lbann::protobuf::as_unordered_set<lbann::protobuf::uint32>(
            msg,
            "my_uint32s") ==
          std::unordered_set<lbann::protobuf::uint32>{1U, 2U, 3U});
    CHECK(
      lbann::protobuf::as_unordered_set<lbann::protobuf::int64>(msg,
                                                                "my_int64s") ==
      std::unordered_set<lbann::protobuf::int64>{16L, 32L, 64L});

    // Enums can be dynamically fetched as int32s or by their enum type.
    CHECK(lbann::protobuf::as_unordered_set<lbann::protobuf::int32>(
            msg,
            "my_enums") == std::unordered_set<lbann::protobuf::int32>{0, 1, 2});
    CHECK(
      lbann::protobuf::as_unordered_set<lbann_testing::MyEnum>(msg,
                                                               "my_enums") ==
      std::unordered_set<lbann_testing::MyEnum>{lbann_testing::ZERO,
                                                lbann_testing::ONE,
                                                lbann_testing::TWO});
  }
}

TEST_CASE("Repeated string/message field extraction", "[protobuf][utils]")
{
  std::string const msg_repeated_ptr_field = R"ptext(
my_simple_msgs {
  my_int32: 13
  my_float: 42.0
  my_string: "pickles"
}
my_simple_msgs {
  my_int32: 32
  my_float: 64.0
  my_string: "apples"
}
my_simple_msgs {
  my_int32: 13
  my_float: 42.0
  my_string: "pickles"
}
my_strings: "hello"
my_strings: "world"
my_strings: "hello"
)ptext";

  lbann_testing::HasRepeatedPtrFields msg;
  CHECK_NOTHROW(lbann::protobuf::text::fill(msg_repeated_ptr_field, msg));

  SECTION("As vector - static field")
  {
    CHECK(lbann::protobuf::as_vector(msg.my_strings()) ==
          std::vector<std::string>{"hello", "world", "hello"});

    auto msg_vec = lbann::protobuf::as_vector(msg.my_simple_msgs());
    CHECK(msg_vec.size() == 3UL);
    CHECK(msg_vec[0].my_int32() == 13);
    CHECK(msg_vec[0].my_float() == 42.0);
    CHECK(msg_vec[0].my_string() == "pickles");
    CHECK(msg_vec[1].my_int32() == 32);
    CHECK(msg_vec[1].my_float() == 64.0);
    CHECK(msg_vec[1].my_string() == "apples");
    CHECK(msg_vec[2].my_int32() == 13);
    CHECK(msg_vec[2].my_float() == 42.0);
    CHECK(msg_vec[2].my_string() == "pickles");
  }

  SECTION("As vector - dynamic field")
  {
    CHECK(lbann::protobuf::as_vector<std::string>(msg, "my_strings") ==
          std::vector<std::string>{"hello", "world", "hello"});

    auto msg_vec =
      lbann::protobuf::as_vector<lbann_testing::SimpleMesg>(msg,
                                                            "my_simple_msgs");
    CHECK(msg_vec.size() == 3UL);
    CHECK(msg_vec[0].my_int32() == 13);
    CHECK(msg_vec[0].my_float() == 42.0);
    CHECK(msg_vec[0].my_string() == "pickles");
    CHECK(msg_vec[1].my_int32() == 32);
    CHECK(msg_vec[1].my_float() == 64.0);
    CHECK(msg_vec[1].my_string() == "apples");
    CHECK(msg_vec[2].my_int32() == 13);
    CHECK(msg_vec[2].my_float() == 42.0);
    CHECK(msg_vec[2].my_string() == "pickles");

    CHECK(msg_vec[0].SerializeAsString() == msg_vec[2].SerializeAsString());
  }

  SECTION("As set - static field")
  {
    CHECK(lbann::protobuf::as_set(msg.my_strings()) ==
          std::set<std::string>{"hello", "world"});
  }

  SECTION("As set - dynamic field")
  {
    CHECK(lbann::protobuf::as_set<std::string>(msg, "my_strings") ==
          std::set<std::string>{"hello", "world"});
  }

  SECTION("As unordered set - static field")
  {
    // using MsgT = lbann_testing::SimpleMesg;
    // using MsgHasher = lbann::protobuf::UnsafeMsgHasher<MsgT>;
    // using MsgEql = lbann::protobuf::MsgEquals<MsgT>;

    // MsgT m1, m2;
    // m1.set_my_int32(13);
    // m1.set_my_float(42.0);
    // m1.set_my_string("pickles");
    // m1.set_my_int32(32);
    // m1.set_my_float(64.0);
    // m1.set_my_string("apples");

    CHECK(lbann::protobuf::as_unordered_set(msg.my_strings()) ==
          std::unordered_set<std::string>{"hello", "world"});
    // FIXME -- DELETE
    // CHECK(lbann::protobuf::as_unordered_set(msg.my_simple_msgs()) ==
    //       std::unordered_set<MsgT, MsgHasher, MsgEql>{m1, m2});
  }

  SECTION("As unordered set - dynamic field")
  {
    CHECK(lbann::protobuf::as_unordered_set<std::string>(msg, "my_strings") ==
          std::unordered_set<std::string>{"hello", "world"});
    // CHECK(lbann::protobuf::as_unordered_set<int>(msg, "my_strings") ==
    //       std::unordered_set<int>{1, 2});
  }
}
