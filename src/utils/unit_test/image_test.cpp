// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/image.hpp>

// Hide by default because this will create a file.
TEST_CASE("Testing image utils", "[.image-utils][utilities]") {
  SECTION("JPEG") {
    std::string filename = "test.jpg";
    lbann::CPUMat image;
    // Make this a 3-channel image.
    image.Resize(3*32*32, 1);
    {
      lbann::DataType* buf = image.Buffer();
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t col = 0; col < 32; ++col) {
          for (size_t row = 0; row < 32; ++row) {
            const size_t i = channel*32*32 + row+col*32;
            if (row == col) { buf[i] = 1.0f; }
            else { buf[i] = 0.0f; }
          }
        }
      }
    }
    SECTION("save image") {
      std::vector<size_t> dims = {3, 32, 32};
      REQUIRE_NOTHROW(lbann::save_image(filename, image, dims));
    }
    SECTION("load image") {
      El::Matrix<uint8_t> loaded_image;
      std::vector<size_t> dims;
      REQUIRE_NOTHROW(lbann::load_image(filename, loaded_image, dims));
      REQUIRE(dims.size() == 3);
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 32);
      REQUIRE(dims[2] == 32);
      const uint8_t* buf = loaded_image.LockedBuffer();
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t col = 0; col < 32; ++col) {
          for (size_t row = 0; row < 32; ++row) {
            const size_t i = 3*(col+row*32) + channel;
            if (row == col) { REQUIRE(buf[i] == 255); }
            // Turns out JPEG doesn't encode every pixel to exactly 0.
            else { REQUIRE(buf[i] <= 1); }
          }
        }
      }
    }
  }
}
