#include <catch2/catch.hpp>

#include <lbann/proto/proto_common.hpp>

#include <string>

TEST_CASE("Testing string trimming", "[proto][utilities]")
{
  SECTION("Leading spaces")
  {
    CHECK(lbann::trim(" my string") == "my string");
    CHECK(lbann::trim("\nmy string") == "my string");
    CHECK(lbann::trim("\tmy string") == "my string");
    CHECK(lbann::trim(" \n\tmy string") == "my string");
    CHECK(lbann::trim("      my string") == "my string");
  }
  SECTION("Trailing spaces")
  {
    CHECK(lbann::trim("my string ") == "my string");
    CHECK(lbann::trim("my string\n") == "my string");
    CHECK(lbann::trim("my string\t") == "my string");
    CHECK(lbann::trim("my string \n\t") == "my string");
    CHECK(lbann::trim("my string    ") == "my string");
  }
  SECTION("Leading and trailing spaces")
  {
    CHECK(lbann::trim(" my string ") == "my string");
    CHECK(lbann::trim(" my string\n") == "my string");
    CHECK(lbann::trim(" my string\t") == "my string");
    CHECK(lbann::trim(" my string \n\t") == "my string");
    CHECK(lbann::trim(" my string    ") == "my string");

    CHECK(lbann::trim("\nmy string ") == "my string");
    CHECK(lbann::trim("\nmy string\n") == "my string");
    CHECK(lbann::trim("\nmy string\t") == "my string");
    CHECK(lbann::trim("\nmy string \n\t") == "my string");
    CHECK(lbann::trim("\nmy string    ") == "my string");

    CHECK(lbann::trim("\tmy string ") == "my string");
    CHECK(lbann::trim("\tmy string\n") == "my string");
    CHECK(lbann::trim("\tmy string\t") == "my string");
    CHECK(lbann::trim("\tmy string \n\t") == "my string");
    CHECK(lbann::trim("\tmy string    ") == "my string");

    CHECK(lbann::trim(" \n\tmy string ") == "my string");
    CHECK(lbann::trim(" \n\tmy string\n") == "my string");
    CHECK(lbann::trim(" \n\tmy string\t") == "my string");
    CHECK(lbann::trim(" \n\tmy string \n\t") == "my string");
    CHECK(lbann::trim(" \n\tmy string    ") == "my string");

    CHECK(lbann::trim("  my string ") == "my string");
    CHECK(lbann::trim("   my string\n") == "my string");
    CHECK(lbann::trim("    my string\t") == "my string");
    CHECK(lbann::trim("     my string \n\t") == "my string");
    CHECK(lbann::trim("      my string    ") == "my string");
  }
  SECTION("Neither leading nor trailing spaces")
  {
    CHECK(lbann::trim("my string") == "my string");
    CHECK(lbann::trim("lbann") == "lbann");
  }
  SECTION("Only spaces")
  {
    CHECK(lbann::trim(" ") == "");
    CHECK(lbann::trim("\n") == "");
    CHECK(lbann::trim("\t") == "");
    CHECK(lbann::trim(" \n\t") == "");
    CHECK(lbann::trim("     \t\n\t") == "");
  }
  SECTION("Empty string")
  {
    CHECK(lbann::trim("") == "");
  }
}
