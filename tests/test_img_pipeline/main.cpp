#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include "lbann/data_readers/lbann_image_utils.hpp"


using namespace lbann::patchworks;

bool test_image_io(const std::string filename, int sz);

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: > " << argv[0] << " image_filenamei [n]" << std::endl;
        std::cout << "    n: the number of channel values in the images (i.e., width*height*num_channels)" << std::endl;
        return 0;
    }

    std::string filename = argv[1];
    int sz = 0;
    if (argc > 2) {
      sz = atoi(argv[2]);
    }

    // read write test with converting to/from a serialized buffer
    bool ok = test_image_io(filename, sz);
    if (!ok) {
        std::cout << "Test failed" << std::endl;
        return 0;
    }
    std::cout << "Complete!" << std::endl;

    return 0;
}

void show_image_size(const int width, const int height, const int type)
{
    const int depth = CV_MAT_DEPTH(type);
    const int NCh = CV_MAT_CN(type);
    const int esz = CV_ELEM_SIZE(depth);
    std::cout << "Image size                     : " << width << " x " << height << std::endl;
    std::cout << "Number of channels             : " << NCh << std::endl;
    std::cout << "Size of the channel value type : " << esz << std::endl;
    std::cout << "Total bytes                    : " << width*height*NCh*esz << std::endl;
}

std::string get_file_extention(const std::string filename)
{
    size_t pos = filename.find_last_of('.');
    if (pos == 0u) return "";
    return filename.substr(pos+1, filename.size());
}

bool read_file(const std::string filename, std::vector<unsigned char>& buf)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.good()) return false;

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

void write_file(const std::string filename, const std::vector<unsigned char>& buf)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file.write((const char*) buf.data(), buf.size() * sizeof(unsigned char));
    file.close();
}

bool test_image_io(const std::string filename, int sz)
{
    std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
    normalizer->z_score(true);

    lbann::cv_process pp;
    pp.set_normalizer(std::move(normalizer));

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

    { // inbuf scope
      // using a portion of the buf
      typedef cv_image_type<uint8_t, 1> InputBuf_T;
      size_t img_begin = 0u;
      size_t img_end = fsz;
      // construct a view of a portion of the existing data
      const cv::Mat inbuf(1, (img_end-img_begin), InputBuf_T::T(), &(buf[img_begin]));
      std::cout << "address of the zero copying view: "
                << reinterpret_cast<unsigned long long>(reinterpret_cast<const void*>(inbuf.datastart)) << " "
                << reinterpret_cast<unsigned long long>(reinterpret_cast<const void*>(&(buf[img_begin]))) << std::endl;

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
