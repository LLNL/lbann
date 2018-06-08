#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_readers/cv_process.hpp"
#include "lbann/utils/file_utils.hpp"


struct cropper_params {
  bool m_is_set;
  bool m_rand_center;
  bool m_adaptive_interpolation;
  std::pair<int, int> m_crop_sz;
  std::pair<int, int> m_roi_sz;

  cropper_params(void)
    : m_is_set(false),
      m_rand_center(false),
      m_adaptive_interpolation(false),
      m_crop_sz(std::make_pair(0, 0)),
      m_roi_sz(std::make_pair(0,0)) {}
};

struct resizer_params {
  bool m_is_set;
  unsigned int m_width;
  unsigned int m_height;
  bool m_adaptive_interpolation;
  resizer_params(void)
    : m_is_set(false),
      m_width(0u),
      m_height(0u),
      m_adaptive_interpolation(false) {}
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

struct main_params {
  enum normalizer_type {_NONE_,_CHANNEL_WISE_,_PIXEL_WISE_};
  unsigned int m_num_bytes;
  bool m_enable_cropper;
  bool m_enable_resizer;
  bool m_enable_augmenter;
  bool m_enable_colorizer;
  bool m_enable_decolorizer;
  bool m_enable_mean_extractor;
  normalizer_type m_enable_normalizer;
  unsigned int m_mean_batch_size;
  unsigned int m_num_iter;
  std::string m_mean_image_name;

  bool is_normalizer_off() const { return (m_enable_normalizer == _NONE_); }
  bool is_channel_wise_normalizer() const { return (m_enable_normalizer == _CHANNEL_WISE_); }
  bool is_pixel_wise_normalizer() const { return (m_enable_normalizer == _PIXEL_WISE_); }

  main_params(void)
    : m_num_bytes(0u),
      m_enable_cropper(true),
      m_enable_resizer(false),
      m_enable_augmenter(false),
      m_enable_colorizer(false),
      m_enable_decolorizer(false),
      m_enable_mean_extractor(true),
      m_enable_normalizer(_NONE_),
      m_mean_batch_size(1024u),
      m_num_iter(1u) {}
};

bool test_image_io(const std::string filename, const main_params& op, const cropper_params& rp, const resizer_params& sp, const augmenter_params& ap);

void show_help(std::string name);

//-----------------------------------------------------------------------------
int main(int argc, char *argv[]) {

  if (argc != 11) {
    show_help(argv[0]);
    return 0;
  }

  std::string filename = argv[1];

  main_params mp;
  mp.m_enable_cropper = true;
  // to test resizer manually swap m_enalbe_cropper/resizer
  mp.m_enable_resizer = false;
  mp.m_enable_augmenter = static_cast<bool>(atoi(argv[8]));
  mp.m_enable_colorizer = true;
  mp.m_enable_decolorizer = false;
  mp.m_enable_normalizer = static_cast<main_params::normalizer_type>(atoi(argv[9]));
  if (mp.is_pixel_wise_normalizer()) mp.m_mean_image_name = "mean.png";
  mp.m_mean_batch_size = atoi(argv[7]);
  mp.m_enable_mean_extractor = (mp.m_mean_batch_size > 0);
  mp.m_num_iter = atoi(argv[10]);

  cropper_params rp;
  if (mp.m_enable_cropper) {
    rp.m_is_set = true;
    rp.m_crop_sz.first = atoi(argv[2]);
    rp.m_crop_sz.second = atoi(argv[3]);
    rp.m_rand_center = static_cast<bool>(atoi(argv[4]));
    rp.m_roi_sz.first = atoi(argv[5]);
    rp.m_roi_sz.second = atoi(argv[6]);
    //rp.m_adaptive_interpolation = true;
  }

  resizer_params sp;
  if (mp.m_enable_resizer) {
    sp.m_is_set = true;
    sp.m_width = static_cast<unsigned int>(atoi(argv[2]));
    sp.m_height = static_cast<unsigned int>(atoi(argv[3]));
    //sp.m_adaptive_interpolation = true;
  }

  augmenter_params ap;
  if (mp.m_enable_augmenter) {
    ap.m_is_set = true;
    ap.m_rot = 0.1;
    ap.m_shear = 0.2;
    ap.m_vflip = true;
  }

  // read write test with converting to/from a serialized buffer
  bool ok = test_image_io(filename, mp, rp, sp, ap);
  if (!ok) {
    std::cout << "Test failed" << std::endl;
    return 0;
  }
  std::cout << "Complete!" << std::endl;

  return 0;
}

//-----------------------------------------------------------------------------
void show_help(std::string name) {
    std::cout << "Usage: > " << name << " image_filename w h r rw rh bsz a n ni" << std::endl;
    std::cout << std::endl;
    std::cout << "    The parameters w, h, c, rw and rh are for cropper" << std::endl;
    std::cout << "    w: the final crop width of image" << std::endl;
    std::cout << "    h: the final crop height of image" << std::endl;
    std::cout << "       (w and h are dictated whether by cropping images to the size)" << std::endl;
    std::cout << "    r: whether to randomize the crop position within the center region (0|1)" << std::endl;
    std::cout << "   rw: The width of the center region with respect to w after resizig the raw image" << std::endl;
    std::cout << "   rh: The height of the center region with respect to h after resizing the raw image" << std::endl;
    std::cout << "       Raw image will be resized to an image of size rw x rh around the center," << std::endl;
    std::cout << "       which covers area of the original image as much as possible while preseving" << std::endl;
    std::cout << "       the aspect ratio of object in the image" << std::endl;
    std::cout << std::endl;
    std::cout << "  bsz: The batch size for mean extractor" << std::endl;
    std::cout << "       if 0, turns off te mean extractor" << std::endl;
    std::cout << std::endl;
    std::cout << "    a: whether to use augmenter (0|1)" << std::endl;
    std::cout << std::endl;
    std::cout << "    n: whether to use normalizer (0=none|1=channel-wise|2=pixel-wise)" << std::endl;
    std::cout << std::endl;
    std::cout << "   ni: The number of iterations." << std::endl;
    std::cout << "       must be greater than 0" << std::endl;
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

void write_file(const std::string filename, const std::vector<unsigned char>& buf) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  file.write((const char *) buf.data(), buf.size() * sizeof(unsigned char));
  file.close();
}

//-----------------------------------------------------------------------------
bool test_image_io(const std::string filename,
  const main_params& mp,
  const cropper_params& rp,
  const resizer_params& sp,
  const augmenter_params& ap)
{

  int transform_idx = 0;
  int mean_extractor_idx = -1;
  unsigned int num_bytes = mp.m_num_bytes; // size of image in bytes

  lbann::cv_process pp;
  { // Initialize the image processor
    if (rp.m_is_set) { // If cropper parameters are given
      // Setup a cropper
      std::unique_ptr<lbann::cv_cropper> cropper(new(lbann::cv_cropper));
      cropper->set(rp.m_crop_sz.first, rp.m_crop_sz.second, rp.m_rand_center, rp.m_roi_sz, rp.m_adaptive_interpolation);
      pp.add_transform(std::move(cropper));
      num_bytes = rp.m_crop_sz.first * rp.m_crop_sz.second * 3;
      transform_idx ++;
    }

    if (sp.m_is_set) { // If resizer parameters are given
      // Setup a cropper
      std::unique_ptr<lbann::cv_resizer> resizer(new(lbann::cv_resizer));
      resizer->set(sp.m_width, sp.m_height, rp.m_adaptive_interpolation);
      pp.add_transform(std::move(resizer));
      num_bytes = sp.m_width * sp.m_height * 3;
      transform_idx ++;
    }

    if (ap.m_is_set) { // Set up an augmenter
      std::unique_ptr<lbann::cv_augmenter> augmenter(new(lbann::cv_augmenter));
      augmenter->set(ap.m_hflip, ap.m_vflip, ap.m_rot, ap.m_hshift, ap.m_vshift, ap.m_shear);
      pp.add_transform(std::move(augmenter));
      transform_idx ++;
    }

    if (mp.m_enable_colorizer) { // Set up a colorizer
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      pp.add_transform(std::move(colorizer));
      transform_idx ++;
    }

    if (mp.m_enable_decolorizer) { // Set up a colorizer
      std::unique_ptr<lbann::cv_decolorizer> decolorizer(new(lbann::cv_decolorizer));
      pp.add_transform(std::move(decolorizer));
      transform_idx ++;
    }

    if (mp.m_enable_mean_extractor) { // set up a mean extractor
      mean_extractor_idx = transform_idx;
      std::unique_ptr<lbann::cv_mean_extractor> mean_extractor(new(lbann::cv_mean_extractor));
      if (rp.m_is_set)
        mean_extractor->set(rp.m_crop_sz.first, rp.m_crop_sz.second, 3, mp.m_mean_batch_size);
      else
        mean_extractor->set(mp.m_mean_batch_size);
      pp.add_transform(std::move(mean_extractor));
      transform_idx ++;
    }

    if (!mp.is_normalizer_off()) { // Set up a normalizer
      if (mp.is_channel_wise_normalizer()) {
        std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
        normalizer->z_score(true);
        pp.add_normalizer(std::move(normalizer));
      } else {
        std::unique_ptr<lbann::cv_subtractor> normalizer(new(lbann::cv_subtractor));
#if 0
        cv::Mat img_to_sub = cv::imread(mp.m_mean_image_name);
        if (img_to_sub.empty()) {
          std::cout << mp.m_mean_image_name << " does not exist" << std::endl;
          return false;
        }
        normalizer->set_mean(img_to_sub);
#else
        std::vector<lbann::DataType> mean = {0.40625, 0.45703, 0.48047};
        normalizer->set_mean(mean);
        std::vector<lbann::DataType> stddev = {0.3, 0.5, 0.3};
        normalizer->set_stddev(stddev);
#endif
        pp.add_normalizer(std::move(normalizer));
      }
      transform_idx ++;
    }
  }

  // Load an image bytestream into memory
  std::vector<unsigned char> buf;
  bool ok = lbann::load_file(filename, buf);
  if (!ok) {
    std::cout << "Failed to load" << std::endl;
    return false;
  }

  int width = 0;
  int height = 0;
  int type = 0;

  ::Mat Images;
  ::Mat Image_v; // matrix view
  Images.Resize(((num_bytes==0)? 1: num_bytes), 2); // minibatch

  size_t img_begin = 0;
  size_t img_end = buf.size();
  for (unsigned int i=0; i < mp.m_num_iter; ++i)
  {
    // This has nothing to do with the image type but only to create view on a block of bytes
    using InputBuf_T = lbann::cv_image_type<uint8_t>;
    // Construct a zero copying view to a portion of a preloaded data buffer
    const cv::Mat inbuf(1, (img_end - img_begin), InputBuf_T::T(1), &(buf[img_begin]));

    if (num_bytes == 0) {
      ok = lbann::image_utils::import_image(inbuf, width, height, type, pp, Images);
      num_bytes = Images.Height();
      El::View(Image_v, Images, El::IR(0, num_bytes), El::IR(0, 1));
    } else {
      El::View(Image_v, Images, El::IR(0, num_bytes), El::IR(0, 1));
      //ok = lbann::image_utils::import_image(buf, width, height, type, pp, Image_v);
      ok = lbann::image_utils::import_image(inbuf, width, height, type, pp, Image_v);
    }
    if (!ok) {
      std::cout << "Failed to import" << std::endl;
      return false;
    }
    //if ((i%3 == 0u) && (mp.m_enable_mean_extractor)) {
    //  dynamic_cast<lbann::cv_mean_extractor*>(pp.get_transform(mean_extractor_idx))->reset();
    //}
  }

  // Print out transforms
  const unsigned int num_transforms = pp.get_num_transforms();
  const std::vector<std::unique_ptr<lbann::cv_transform> >& transforms = pp.get_transforms();

  for(unsigned int i=0u; i < num_transforms; ++i) {
    std::cout << std::endl << "------------ transform " << i << "-------------" << std::endl;
    std::cout << *transforms[i] << std::endl;
  }

  if (mp.m_enable_mean_extractor) {
    // Extract the mean of images
    cv::Mat mean_image;
    mean_image = dynamic_cast<lbann::cv_mean_extractor*>(pp.get_transform(mean_extractor_idx))->extract<uint16_t>();
    cv::imwrite("mean.png", mean_image);
  }

  // Export the unnormalized image
  const std::string ext = lbann::get_ext_name(filename);
  std::vector<unsigned char> outbuf;
  ok = lbann::image_utils::export_image(ext, outbuf, width, height, type, pp, Image_v);
  write_file("copy." + ext, outbuf);
  return ok;
}
