#include <algorithm>
#include <iterator>
#include <sstream>
#include <ctime>
#include <cmath> // sqrt
#include <cstdlib>
#include "lbann/data_readers/patchworks/patchworks_stats.hpp"
#include "patchworks_image.hpp"

#include <opencv2/core/version.hpp>
#if (!defined(CV_VERSION_EPOCH) && (CV_VERSION_MAJOR >= 3))
  #define DEFAULT_CV_WINDOW_KEEPRATIO cv::WINDOW_KEEPRATIO
#else
  #define DEFAULT_CV_WINDOW_KEEPRATIO CV_WINDOW_KEEPRATIO
#endif

namespace lbann {
namespace patchworks {

std::string showDepth(const cv::Mat& mat)
{
    return showDepth(CV_MAT_DEPTH(mat.type()));
}

std::string showDepth(const int depth)
{
    switch (depth) {
        case CV_8U: return "CV_8U";   break;
        case CV_8S: return "CV_8S";   break;
        case CV_16U: return "CV_16U"; break;
        case CV_16S: return "CV_16S"; break;
        case CV_32S: return "CV_32S"; break;
        case CV_32F: return "CV_32F"; break;
        case CV_64F: return "CV_64F"; break;
        default: return "Unknown"; break;
    }
    return "Unknown";
}

size_t image_data_amount(const cv::Mat& img)
{
    return static_cast<size_t>(CV_ELEM_SIZE(img.depth())*CV_MAT_CN(img.type())*img.cols*img.rows);
}

void show_cvMat_info(const int type)
{
    const int depth = CV_MAT_DEPTH(type);
    std::cout << "showDepth(CV_MAT_DEPTH(img.type())) " << lbann::patchworks::showDepth(depth) << std::endl;
    std::cout << "CV_ELEM_SIZE(img.depth()) " << CV_ELEM_SIZE(depth) << std::endl;
    std::cout << "CV_ELEM_SIZE(CV_MAT_DEPTH(img.type())) " << CV_ELEM_SIZE(depth) << std::endl;
    std::cout << "CV_MAT_CN(img.type())    " << CV_MAT_CN(type) << std::endl;
}


image::image(const std::string fname)
: m_screen_width(640), m_screen_height(480)
{
  detect_screen_resolution();
  if (fname != "") load(fname);
}

image::~image(void)
{
  release();
}

void image::release(void)
{
  m_filename = "";
  if (m_img.data != NULL) m_img.release();
  m_img.data = NULL;
}

bool image::load(const std::string fname)
{
  release();

  m_img = cv::imread(fname, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

  if (m_img.data == NULL) return false;

  m_filename = fname;
  return true;
}


std::ostream& image::show_info(const cv::Mat& img, const std::string title, std::ostream& os)
{
  if (img.data == NULL)  return os;

  os << title << std::endl
     << "   Type: " << showDepth(img) << std::endl
     << "   nCh : " << img.channels() << std::endl
     << "   Size: " << img.size() << std::endl;

  std::vector<image_stats> stats;
  get_channel_stats(img, stats);

  for (int ch = 0; ch < img.channels(); ++ch) {
    os << "channel " << ch;
    os << stats[ch] << std::endl;
  }

  return os;
}

std::ostream& image::show_info(const std::string title, std::ostream& os) const
{
  show_info(m_img, title, os);
  os << "   File: " << m_filename << std::endl;
  return os;
}

void image::detect_screen_resolution(void)
{
  std::vector<std::pair<int, int> > res;
  unsigned int cnt = get_screen_resolution(res);

  if (cnt == 0u) { // fall back resolution
    res.push_back(std::make_pair(640,480));
  }

  m_screen_width = res[0].first;
  m_screen_height = res[0].second;
}

void image::display(const std::string title) const
{
  if (m_img.data == NULL) return; // nothing to show

  cv::namedWindow(title, cv::WINDOW_NORMAL | DEFAULT_CV_WINDOW_KEEPRATIO);
  cv::imshow(title, m_img);

  float zoomFactor = 1.0;
  const double initialZoomOutRateW = static_cast<double>(m_screen_width)/m_img.cols;
  const double initialZoomOutRateH = static_cast<double>(m_screen_height)/m_img.rows;
  const double initialZoomOutRate = std::min(initialZoomOutRateW, initialZoomOutRateH);

  const int eW = static_cast<int>(m_img.cols * initialZoomOutRate);
  const int eH = static_cast<int>(m_img.rows * initialZoomOutRate);

  const int m_screen_widthZ = std::min(static_cast<int>(zoomFactor*eW), m_img.cols);
  const int m_screen_heightZ = std::min(static_cast<int>(zoomFactor*eH), m_img.rows);

  cv::resizeWindow(title, m_screen_widthZ, m_screen_heightZ);
  m_window_title = title;
  cv::waitKey(0);
}

void image::draw_rectangle(cv::Mat& img, const ROI& r)
{
  const uint16_t chSet = std::numeric_limits<uint16_t>::max();
  const cv::Scalar color(0, chSet, chSet);
  const int thickness = 2;
  const int lineType = 8;

  cv::rectangle(img,
                cv::Point(r.left(), r.top()),
                cv::Point(r.right(), r.bottom()),
                color, thickness, lineType);
}

void image::draw_rectangle(const ROI& r)
{
  draw_rectangle(m_img, r);
}

void image::draw_patches(const patch_descriptor& pi)
{
  const std::vector<ROI>& pos = pi.access_positions();
  for (size_t i=0u; i< pos.size(); ++i) {
    draw_rectangle(m_img, pos[i]);
  }
}

bool image::write(const std::string out_filename, const cv::Mat& img_to_write)
{
  if ((out_filename == "") || (img_to_write.data == NULL)) {
    return false;
    //std::cout << "Failed to write an image file [" << out_filename << "]" << std::endl;
  }
  //std::cout << "writing an image file [" << out_filename << "]" << std::endl;
  return cv::imwrite(out_filename, img_to_write);
}


} // end of namespace patchworks
} // end of namespace lbann
