/*
 *  Main image data structure
 *  Author: Jae-Seung Yeom
 */

#ifndef _PATCHWORKS_IMAGE_H_INCLUDED_
#define _PATCHWORKS_IMAGE_H_INCLUDED_

#include <string>
#include <ostream>
#include <iostream>
#include <vector>
#include "lbann/data_readers/patchworks/patchworks_common.hpp"
#include "lbann/data_readers/patchworks/patchworks_ROI.hpp"
#include "lbann/data_readers/patchworks/patchworks_patch_descriptor.hpp"
#include "patchworks_utils.hpp"

#include <opencv2/core/version.hpp>
#if (!defined(CV_VERSION_EPOCH) && (CV_VERSION_MAJOR >= 3))
#include <opencv2/highgui.hpp>
#define DEFAULT_CV_WINDOW_KEEPRATIO cv::WINDOW_KEEPRATIO
#else
#include <opencv2/highgui/highgui.hpp>
#define DEFAULT_CV_WINDOW_KEEPRATIO CV_WINDOW_KEEPRATIO
#endif

namespace lbann {
namespace patchworks {

class image {
 public:
 protected:
  /// The image data (OpenCV Mat type)
  cv::Mat m_img;
  /// A name to show on an image window title bar
  mutable std::string m_window_title;
  /// The name of the image file
  std::string m_filename;

  // current screen resolution
  int m_screen_width; ///< screen width in the number of pixels
  int m_screen_height; ///< screen height in the number of pixels

  /// Detect screen resolution to draw window that fits in the screen
  virtual void detect_screen_resolution(void);

 public:
  image(void) : m_window_title(""), m_filename("") {
    detect_screen_resolution();
  }
  image(const std::string fname);
  virtual ~image(void);

  /// Check if an image data exists
  virtual bool empty(void) const {
    return (m_img.data == NULL);
  }
  /// Free the space used by image data
  virtual void release(void);
  /// Return the filename of this image
  virtual std::string get_filename(void) const {
    return m_filename;
  }
  /// Read an image file
  virtual bool load(const std::string fname);
  /// Display the image
  virtual void display(const std::string title="") const;
  /// Show information on a given image with a given title
  static std::ostream& show_info(const cv::Mat& img, const std::string title = "image info",
                                 std::ostream& os = std::cout);
  /// Show information on the image with a given title
  virtual std::ostream& show_info(const std::string title = "image info",
                                  std::ostream& os = std::cout) const;

  /// Return the width (number of columns) of the image
  virtual int get_width(void) const {
    return m_img.cols;
  }
  /// Return the height (number of rows) of the image
  virtual int get_height(void) const {
    return m_img.rows;
  }

  /// Return the number of channels of the images
  virtual int get_num_channels(void) const {
    return m_img.channels();
  }
  /// Return the pixel depth of the image in OpenCV term
  virtual int get_depth(void) const {
    return m_img.depth();
  }

  /// Returns the access to the image data (OpenCV Mat type)
  virtual cv::Mat& get_image(void) {
    return m_img;
  }

  /// Mark a retangular region on the image
  static void draw_rectangle(cv::Mat& img, const ROI& r);
  /// Mark a retangular region on the image
  virtual void draw_rectangle(const ROI& r);
  /// Mark patch regions on the image
  virtual void draw_patches(const patch_descriptor& pi);

  /// Write an image into the file with the given file name
  static bool write(const std::string outFileName, const cv::Mat& img_to_write);
  /// Write the image into the file with the given name
  virtual bool write(const std::string outFileName) const {
    return write(outFileName, m_img);
  }
};

std::string showDepth(const cv::Mat& mat);
std::string showDepth(const int depth);
size_t image_data_amount(const cv::Mat& mat);
void show_cvMat_info(const int type);

} // end of namespace patchworks
} // end of namespace lbann
#endif // _PATCHWORKS_IMAGE_H_INCLUDED_
