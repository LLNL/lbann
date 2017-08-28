#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include "lbann/base.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/data_readers/patchworks/patchworks.hpp"
#include "patchworks_image.hpp"


using namespace lbann::patchworks;

bool test_patch(const int argc, char *argv[]);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: > " << argv[0] << " filename [patch_size [gap_size [jitter [ceteringMode [ca_mode]]]]]" << std::endl;
    return 0;
  }

  std::string filename = argv[1];

  lbann::init_random();
  bool ok = test_patch(argc, argv);
  if (!ok) {
    std::cout << "failed to copy the image" << std::endl;
    return 0;
  }
  std::cout << "Complete!" << std::endl;

  return 0;
}

bool test_patch(const int argc, char *argv[]) {
  unsigned int patch_size = 96u;
  unsigned int gap = 48u;
  unsigned int jitter = 7u;
  unsigned int mode_centering = 1u;
  unsigned int mode_chromaberr = 0u;
  std::string filename = argv[1];
  if (argc > 2) {
    patch_size = std::atoi(argv[2]);
  }
  if (argc > 3) {
    gap = std::atoi(argv[3]);
  }
  if (argc > 4) {
    jitter = std::atoi(argv[4]);
  }
  if (argc > 5) {
    mode_centering = std::atoi(argv[5]);
  }
  if (argc > 6) {
    mode_chromaberr = std::atoi(argv[6]);
  }

  if (patch_size == 0u) {
    return false;
  }

#ifdef __LIB_OPENCV
  // load input image
  image *img = new image(filename);
  if (img->empty()) {
    std::cout << "failed to load the image " << filename << std::endl;
    return false;
  }
  img->show_info();
  img->display("original " + filename);

  bool ok = true;
  patch_descriptor pi;
  pi.set_size(patch_size, patch_size);
  ok = pi.set_sample_image(static_cast<unsigned int>(img->get_width()),
                           static_cast<unsigned int>(img->get_height()));
  if (!ok) {
    std::cout << "failed to set patch sampling region" << std::endl;
  }
  pi.set_gap(gap);
  pi.set_jitter(jitter);
  pi.set_mode_centering(mode_centering);
  pi.set_mode_chromatic_aberration(mode_chromaberr);
  pi.set_file_ext("png");
  pi.define_patch_set();

  std::vector<cv::Mat> patches;
  ok = pi.extract_patches(img->get_image(), patches);
  if (!ok) {
    std::cout << "failed to extract patch" << std::endl;
  }
  for (size_t i=0u; i < patches.size(); ++i) {
    std::stringstream sstr;
    sstr << "patch." << i << ".png";
    image::write(sstr.str(), patches[i]);
  }
  std::cout << "the id of the last patch generated (label in case of paired patches): "
            << pi.get_current_patch_idx() << std::endl;

  std::cout << pi;
  img->draw_patches(pi);
  img->display("patches of " + filename);

  std::string patched_filename = basename_with_no_extention(filename)
                                 + ".patched." + get_file_extention(filename);
  ok = img->write(patched_filename);
  if (!ok) {
    std::cout << "failed to write patch map" << std::endl;
  }

  delete img;
  return true;
#else
  return false;
#endif // __LIB_OPENCV
}
