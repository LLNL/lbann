#ifndef LBANN_TRANSFORMS_VISION_UNIT_TEST_HELPER
#define LBANN_TRANSFORMS_VISION_UNIT_TEST_HELPER

inline void apply_elementwise(
  El::Matrix<uint8_t>& mat, El::Int height, El::Int width, El::Int channels,
  std::function<void(uint8_t&, El::Int, El::Int, El::Int)> f) {
  uint8_t* buf = mat.Buffer();
  for (El::Int channel = 0; channel < channels; ++channel) {
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        f(buf[channels*(col+row*width) + channel], row, col, channel);
      }
    }
  }
}

inline void identity(El::Matrix<uint8_t>& mat, El::Int height, El::Int width,
              El::Int channels = 1) {
  mat.Resize(height*width*channels, 1);
  apply_elementwise(mat, height, width, channels,
                    [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                      x = (row == col) ? 1 : 0;
                    });
}

inline void zeros(El::Matrix<uint8_t>& mat, El::Int height, El::Int width,
           El::Int channels = 1) {
  mat.Resize(height*width*channels, 1);
  uint8_t* buf = mat.Buffer();
  for (El::Int i = 0; i < height*width*channels; ++i) {
    buf[i] = 0;
  }
}

inline void ones(El::Matrix<uint8_t>& mat, El::Int height, El::Int width,
           El::Int channels = 1) {
  mat.Resize(height*width*channels, 1);
  uint8_t* buf = mat.Buffer();
  for (El::Int i = 0; i < height*width*channels; ++i) {
    buf[i] = 1;
  }
}

inline void print(const El::Matrix<uint8_t>& mat, El::Int height, El::Int width,
           El::Int channels = 1) {
  const uint8_t* buf = mat.LockedBuffer();
  for (El::Int channel = 0; channel < channels; ++channel) {
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        std::cout << ((int) buf[channels*(col+row*width) + channel]) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "--" << std::endl;
  }
}

#endif  // LBANN_TRANSFORMS_VISION_UNIT_TEST_HELPER
