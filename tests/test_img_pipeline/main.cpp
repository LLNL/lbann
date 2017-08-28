#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_readers/cv_resizer.hpp"
#include "lbann/data_readers/cv_colorizer.hpp"


using namespace lbann::patchworks;

struct resizer_params {
  bool m_is_set;
  std::pair<int, int> m_desired_sz;
  std::pair<float, float> m_center;
  std::pair<int, int> m_roi_sz;

  resizer_params(void)
    : m_is_set(false),
      m_desired_sz(std::make_pair(400, 400)),
      m_center(std::make_pair(0.5, 0.5)),
      m_roi_sz(std::make_pair(0,0)) {}
};

struct augmenter_params {
  bool m_is_set;
  bool m_hflip;
  bool m_vflip;
  float m_rot;
  float m_hshift;
  float m_vshift;
  float m_shear;

  augmenter_params(void)
    : m_is_set(false),
      m_hflip(false),
      m_vflip(false),
      m_rot(0.0f),
      m_hshift(0.0f),
      m_vshift(0.0f),
      m_shear(0.0f) {}
};


bool test_image_io(const std::string filename, int sz, const resizer_params& rp, const augmenter_params& ap, const bool do_colorize);


int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Usage: > " << argv[0] << " image_filename [n]|[w h xc yc rw rh]" << std::endl;
    std::cout << "    n: the number of channel values in the images (i.e., width*height*num_channels)" << std::endl;
    std::cout << "    w: the desired width of image" << std::endl;
    std::cout << "    h: the desired height of image" << std::endl;
    std::cout << "   xc: the fractional horizontal position of the desired center in the image" << std::endl;
    std::cout << "   yc: the fractional vertical position of the desired center in the image" << std::endl;
    std::cout << "   rw: the width of the region of interest" << std::endl;
    std::cout << "   rh: the height of the region of interest" << std::endl;
    return 0;
  }

  std::string filename = argv[1];
  int sz = 0;
  resizer_params rp;

  if (argc == 3) {
    sz = atoi(argv[2]);
  } else if (argc == 8) {
    rp.m_desired_sz.first = atoi(argv[2]);
    rp.m_desired_sz.second = atoi(argv[3]);
    rp.m_center.first = atof(argv[4]);
    rp.m_center.second = atof(argv[5]);
    rp.m_roi_sz.first = atoi(argv[6]);
    rp.m_roi_sz.second = atoi(argv[7]);
    rp.m_is_set = true;
  }

  augmenter_params ap;

  // read write test with converting to/from a serialized buffer
  bool ok = test_image_io(filename, sz, rp, ap, false);
  if (!ok) {
    std::cout << "Test failed" << std::endl;
    return 0;
  }
  std::cout << "Complete!" << std::endl;

  return 0;
}

void show_image_size(const int width, const int height, const int type) {
  const int depth = CV_MAT_DEPTH(type);
  const int NCh = CV_MAT_CN(type);
  const int esz = CV_ELEM_SIZE(depth);
  std::cout << "Image size                     : " << width << " x " << height << std::endl;
  std::cout << "Number of channels             : " << NCh << std::endl;
  std::cout << "Size of the channel value type : " << esz << std::endl;
  std::cout << "Total bytes                    : " << width *height *NCh *esz << std::endl;
}

std::string get_file_extention(const std::string filename) {
  size_t pos = filename.find_last_of('.');
  if (pos == 0u) {
    return "";
  }
  return filename.substr(pos+1, filename.size());
}

bool read_file(const std::string filename, std::vector<unsigned char>& buf) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    return false;
  }

  file.unsetf(std::ios::skipws);

  file.seekg(0, std::ios::end);
  const std::streampos file_size = file.tellg();

  file.seekg(0, std::ios::beg);

  buf.reserve(file_size);

  buf.insert(buf.begin(),
             std::istream_iterator<unsigned char>(file),
             std::istream_iterator<unsigned char>());

  return true;
}

void write_file(const std::string filename, const std::vector<unsigned char>& buf) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  file.write((const char *) buf.data(), buf.size() * sizeof(unsigned char));
  file.close();
}

bool test_image_io(const std::string filename, int sz, const resizer_params& rp, const augmenter_params& ap, const bool do_colorize) {
#if 1
  std::shared_ptr<lbann::cv_process> pp1;

  // Initialize the image processor
  // while testing the scope of transform objects managed by smart pointers
  {
    std::shared_ptr<lbann::cv_process> pp0 = std::make_shared<lbann::cv_process>();

    if (rp.m_is_set) { // If resizer parameters are given
      // Setup a resizer
      std::unique_ptr<lbann::cv_resizer> resizer(new(lbann::cv_resizer));
      resizer->set(rp.m_desired_sz.first, rp.m_desired_sz.second, true, rp.m_center, rp.m_roi_sz);
      pp0->set_custom_transform1(std::move(resizer));
      sz = rp.m_desired_sz.first * rp.m_desired_sz.second * 3;
    }

    if (ap.m_is_set) { // If augmenter parameters are given
      // Set up an augmenter
      std::unique_ptr<lbann::cv_augmenter> augmenter(new(lbann::cv_augmenter));
      augmenter->set(ap.m_hflip, ap.m_vflip, ap.m_rot, ap.m_hshift, ap.m_vshift, ap.m_shear);
      pp0->set_augmenter(std::move(augmenter));
    }

    if (do_colorize) {
      // Set up a colorizer
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      pp0->set_custom_transform2(std::move(colorizer));
    }

    // Set up a normalizer
    std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
    normalizer->z_score(true);
    pp0->set_normalizer(std::move(normalizer));

    pp1 = pp0;
  }
  lbann::cv_process pp(*pp1); // testing copy-constructor
#else
  lbann::cv_process pp;
  std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
  normalizer->z_score(true);
  pp.set_normalizer(std::move(normalizer));
#endif

  std::vector<unsigned char> buf;
  bool ok = read_file(filename, buf);
  if (!ok) {
    std::cout << "Failed to load" << std::endl;
    return false;
  }
  const unsigned int fsz = buf.size();
  std::cout << "file size " << fsz << std::endl;


  int width = 0;
  int height = 0;
  int type = 0;
  const int minibatch_size = 2;

  ::Mat Images; // minibatch
  ::Mat Image_v0; // a submatrix view
  ::Mat Image_v1; // a submatrix view

  {
    // inbuf scope
    // using a portion of the buf
    typedef cv_image_type<uint8_t, 1> InputBuf_T;
    size_t img_begin = 0u;
    size_t img_end = fsz;
    // construct a view of a portion of the existing data
    const cv::Mat inbuf(1, (img_end-img_begin), InputBuf_T::T(), &(buf[img_begin]));
    std::cout << "address of the zero copying view: "
              << reinterpret_cast<unsigned long long>(reinterpret_cast<const void *>(inbuf.datastart)) << " "
              << reinterpret_cast<unsigned long long>(reinterpret_cast<const void *>(&(buf[img_begin]))) << std::endl;

    if (sz > 0) {
      std::cout << "The size of the image is as given : " << sz << std::endl;
      // Assuming image croping in the preprocessing pipeline, we know the size of the image,
      // Suppose the size of the image is as give in the command line argument.
      // Then, pass one of the views instead of whole 'Images'
      Images.Resize(sz, minibatch_size); // minibatch
      View(Image_v0, Images, 0, 0, sz, 1);
      View(Image_v1, Images, 0, 1, sz, 1);
      ok = lbann::image_utils::import_image(inbuf, width, height, type, pp, Image_v0);
      if (width*height*CV_MAT_CN(type) != sz) {
        std::cout << "The size of image is not as expected." << std::endl;
        sz = width*height*CV_MAT_CN(type);
        Images.Resize(sz, Images.Width());
      }
    } else {
      std::cout << "We do not know the size of the image yet." << std::endl;
      const int minimal_data_len = 1;
      Images.Resize(minimal_data_len, minibatch_size); // minibatch
      ok = lbann::image_utils::import_image(inbuf, width, height, type, pp, Images);
      sz = Images.Height();
      std::cout << "The size of the image discovered : " << sz << std::endl;
      View(Image_v0, Images, 0, 0, sz, 1);
      View(Image_v1, Images, 0, 1, sz, 1);
    }
  }

  if (!ok) {
    std::cout << "Failed to import" << std::endl;
    return false;
  }
  show_image_size(width, height, type);

  if (pp.custom_transform1()) {
    std::cout << std::endl << "resizing method: "<< std::endl;
    std::cout << *(pp.custom_transform1()) << std::endl;
  }

  Image_v1 = Image_v0;

  std::cout << "Minibatch matrix size: " << Images.Height() << " x " << Images.Width() << std::endl;

  pp.disable_transforms();

  // Write an image
  const std::string ext = get_file_extention(filename);
  pp.determine_inverse_normalization();
  std::vector<unsigned char> outbuf;
  ok = lbann::image_utils::export_image(ext, outbuf, width, height, type, pp, Image_v1);
  write_file("copy." + ext, outbuf);
  return ok;
}
