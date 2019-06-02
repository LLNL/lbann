/*
 *  Utility routines
 *  Author: Jae-Seung Yeom
 */
#ifndef _PATCHWORKS_UTILS_H_INCLUDED_
#define _PATCHWORKS_UTILS_H_INCLUDED_
#include <vector>
#include <utility> // std::pair
#include "lbann/data_readers/patchworks/patchworks_common.hpp"

/// Obtain the screen resolution, which is useful to size an window
unsigned int get_screen_resolution(std::vector<std::pair<int, int> >& res);

/// Split a file path into the directory and the file name under it
bool split_path(const std::string& path, std::string& dir, std::string& name);

/// return the file name without extension
std::string name_with_no_extension(const std::string filename);

/// return the file extension
std::string get_file_extension(const std::string filename);

/// return the base file name (respective to its final directory) without extension
std::string basename_with_no_extension(const std::string filename);

#endif // _PATCHWORKS_UTILS_H_INCLUDED_
