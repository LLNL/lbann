include(ExternalProject)

# Options
option(FORCE_OPENCV_BUILD "OpenCV: force build" OFF)

# Try finding OpenCV
if(NOT FORCE_OPENCV_BUILD)
  find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
endif()

# Check if OpenCV has been found
if(OpenCV_FOUND AND NOT FORCE_OPENCV_BUILD)

  # Status message
  message(STATUS "Found OpenCV (version ${OpenCV_VERSION}): ${OpenCV_DIR}")

else()

  # Git repository URL and tag
  if(NOT OPENCV_URL)
    set(OPENCV_URL https://github.com/opencv/opencv.git)
  endif()
  if(NOT OPENCV_TAG)
    set(OPENCV_TAG "2.4.13")
  endif()
  message(STATUS "Will pull OpenCV (tag ${OPENCV_TAG}) from ${OPENCV_URL}")

  # OpenCV build options
  if(NOT OPENCV_BUILD_TYPE)
    set(OPENCV_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  endif()

  # whether to link with prebuilt libjpeg-turbo
  if(WITH_LIBJPEG_TURBO)
    if ("${LIBJPEG_TURBO_DIR}" STREQUAL "")
      #set(WITH_LIBJPEG_TURBO OFF)
      include(jpeg-turbo)
      set(BUILD_JPEG_TURBO ON)
      message(STATUS "Building libjpeg-turbo")
      set(LIBJPEG_TURBO_DIR                ${CMAKE_INSTALL_PREFIX})
      set(OPENCV_JPEG_INCLUDE_DIR          "${LIBJPEG_TURBO_DIR}/include")
      set(OPENCV_JPEG_LIBRARY              ${LIBJPEG_TURBO_DIR}/lib/libjpeg.so)
      set(CMAKE_LIBRARY_PATH               ${LIBJPEG_TURBO_DIR} ${CMAKE_LIBRARY_PATH})
      set(CMAKE_INCLUDE_PATH               ${OPENCV_JPEG_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
      set(OPENCV_LIBJPEG_TURBO_INC         "-D JPEG_INCLUDE_DIR=${OPENCV_JPEG_INCLUDE_DIR}")
      set(OPENCV_LIBJPEG_TURBO_LIB         "-D JPEG_LIBRARY=${OPENCV_JPEG_LIBRARY}")
    else()
      set(BUILD_JPEG_TURBO OFF)
      message(STATUS "Using libjpeg-turbo installed under ${LIBJPEG_TURBO_DIR}")
      set(OPENCV_JPEG_INCLUDE_DIR          "${LIBJPEG_TURBO_DIR}/include")
      set(OPENCV_JPEG_LIBRARY              ${LIBJPEG_TURBO_DIR}/lib/libjpeg.so)
      set(CMAKE_LIBRARY_PATH               ${LIBJPEG_TURBO_DIR} ${CMAKE_LIBRARY_PATH})
      set(CMAKE_INCLUDE_PATH               ${OPENCV_JPEG_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
      set(OPENCV_LIBJPEG_TURBO_INC         "-D JPEG_INCLUDE_DIR=${OPENCV_JPEG_INCLUDE_DIR}")
      set(OPENCV_LIBJPEG_TURBO_LIB         "-D JPEG_LIBRARY=${OPENCV_JPEG_LIBRARY}")
    endif()
  endif()

  # ============================
  # OpenCV CMake options
  # ============================

  # OpenCV modules
  option(OPENCV_BUILD_opencv_calib3d    "OpenCV: Include opencv_calib3d module into the OpenCV build"                OFF)
  option(OPENCV_BUILD_opencv_contrib    "OpenCV: Include opencv_contrib module into the OpenCV build"                OFF)
  option(OPENCV_BUILD_opencv_core       "OpenCV: Include opencv_core module into the OpenCV build"                   ON)
  option(OPENCV_BUILD_opencv_dynamicuda "OpenCV: Include opencv_dynamicuda module into the OpenCV build"             OFF)
  option(OPENCV_BUILD_opencv_features2d "OpenCV: Include opencv_features2d module into the OpenCV build"             OFF)
  option(OPENCV_BUILD_opencv_flann      "OpenCV: Include opencv_flann module into the OpenCV build"                  OFF)
  option(OPENCV_BUILD_opencv_gpu        "OpenCV: Include opencv_gpu module into the OpenCV build"                    OFF)
  option(OPENCV_BUILD_opencv_highgui    "OpenCV: Include opencv_highgui module into the OpenCV build"                ON)
  option(OPENCV_BUILD_opencv_imgproc    "OpenCV: Include opencv_imgproc module into the OpenCV build"                ON)
  option(OPENCV_BUILD_opencv_java       "OpenCV: Include opencv_java module into the OpenCV build"                   OFF)
  option(OPENCV_BUILD_opencv_legacy     "OpenCV: Include opencv_legacy module into the OpenCV build"                 OFF)
  option(OPENCV_BUILD_opencv_ml         "OpenCV: Include opencv_ml module into the OpenCV build"                     OFF)
  option(OPENCV_BUILD_opencv_nonfree    "OpenCV: Include opencv_nonfree module into the OpenCV build"                OFF)
  option(OPENCV_BUILD_opencv_objdetect  "OpenCV: Include opencv_objdetect module into the OpenCV build"              OFF)
  option(OPENCV_BUILD_opencv_ocl        "OpenCV: Include opencv_ocl module into the OpenCV build"                    OFF)
  option(OPENCV_BUILD_opencv_photo      "OpenCV: Include opencv_photo module into the OpenCV build"                  OFF)
  option(OPENCV_BUILD_opencv_python     "OpenCV: Include opencv_python module into the OpenCV build"                 OFF)
  option(OPENCV_BUILD_opencv_stitching  "OpenCV: Include opencv_stitching module into the OpenCV build"              OFF)
  option(OPENCV_BUILD_opencv_superres   "OpenCV: Include opencv_superres module into the OpenCV build"               OFF)
  option(OPENCV_BUILD_opencv_ts         "OpenCV: Include opencv_ts module into the OpenCV build"                     OFF)
  option(OPENCV_BUILD_opencv_video      "OpenCV: Include opencv_video module into the OpenCV build"                  OFF)
  option(OPENCV_BUILD_opencv_videostab  "OpenCV: Include opencv_videostab module into the OpenCV build"              OFF)
  option(OPENCV_BUILD_opencv_viz        "OpenCV: Include opencv_viz module into the OpenCV build"                    OFF)
  option(OPENCV_BUILD_opencv_world      "OpenCV: Include opencv_world module into the OpenCV build"                  OFF)

  # Optional 3rd party components
  option(OPENCV_WITH_1394           "OpenCV: Include IEEE1394 support"                                                    OFF)
  option(OPENCV_WITH_AVFOUNDATION   "OpenCV: Use AVFoundation for Video I/O (iOS/Mac)"                                    OFF)
  option(OPENCV_WITH_CARBON         "OpenCV: Use Carbon for UI instead of Cocoa"                                          OFF)
  option(OPENCV_WITH_CUDA           "OpenCV: Include NVidia Cuda Runtime support"                                         OFF)
  option(OPENCV_WITH_VTK            "OpenCV: Include VTK library support (and build opencv_viz module eiher)"             OFF)
  option(OPENCV_WITH_CUFFT          "OpenCV: Include NVidia Cuda Fast Fourier Transform (FFT) library support"            OFF)
  option(OPENCV_WITH_CUBLAS         "OpenCV: Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support" OFF)
  option(OPENCV_WITH_NVCUVID        "OpenCV: Include NVidia Video Decoding library support"                               OFF)
  option(OPENCV_WITH_EIGEN          "OpenCV: Include Eigen2/Eigen3 support"                                               OFF)
  option(OPENCV_WITH_VFW            "OpenCV: Include Video for Windows support"                                           OFF)
  option(OPENCV_WITH_FFMPEG         "OpenCV: Include FFMPEG support"                                                      OFF)
  option(OPENCV_WITH_GSTREAMER      "OpenCV: Include Gstreamer support"                                                   OFF)
  option(OPENCV_WITH_GSTREAMER_0_10 "OpenCV: Enable Gstreamer 0.10 support (instead of 1.x)"                              OFF)
  option(OPENCV_WITH_GTK            "OpenCV: Include GTK support"                                                         OFF)
  option(OPENCV_WITH_IMAGEIO        "OpenCV: ImageIO support for OS X"                                                    OFF)
  option(OPENCV_WITH_IPP            "OpenCV: Include Intel IPP support"                                                   OFF) # Causes a hash mismatch error when downloading
  option(OPENCV_WITH_JASPER         "OpenCV: Include JPEG2K support"                                                      OFF)
  option(OPENCV_WITH_JPEG           "OpenCV: Include JPEG support"                                                        ON)
  option(OPENCV_WITH_OPENEXR        "OpenCV: Include ILM support via OpenEXR"                                             OFF)
  option(OPENCV_WITH_OPENGL         "OpenCV: Include OpenGL support"                                                      OFF)
  option(OPENCV_WITH_OPENNI         "OpenCV: Include OpenNI support"                                                      OFF)
  option(OPENCV_WITH_PNG            "OpenCV: Include PNG support"                                                         ON)
  option(OPENCV_WITH_PVAPI          "OpenCV: Include Prosilica GigE support"                                              OFF)
  option(OPENCV_WITH_GIGEAPI        "OpenCV: Include Smartek GigE support"                                                OFF)
  option(OPENCV_WITH_QT             "OpenCV: Build with Qt Backend support"                                               OFF)
  option(OPENCV_WITH_WIN32UI        "OpenCV: Build with Win32 UI Backend support"                                         OFF)
  option(OPENCV_WITH_QUICKTIME      "OpenCV: Use QuickTime for Video I/O"                                                 OFF)
  option(OPENCV_WITH_TBB            "OpenCV: Include Intel TBB support"                                                   OFF)
  option(OPENCV_WITH_OPENMP         "OpenCV: Include OpenMP support"                                                      OFF)
  option(OPENCV_WITH_CSTRIPES       "OpenCV: Include C= support"                                                          OFF)
  option(OPENCV_WITH_TIFF           "OpenCV: Include TIFF support"                                                        ON)
  option(OPENCV_WITH_UNICAP         "OpenCV: Include Unicap support (GPL)"                                                OFF)
  option(OPENCV_WITH_V4L            "OpenCV: Include Video 4 Linux support"                                               OFF)
  option(OPENCV_WITH_LIBV4L         "OpenCV: Use libv4l for Video 4 Linux support"                                        OFF)
  option(OPENCV_WITH_DSHOW          "OpenCV: Build VideoIO with DirectShow support"                                       OFF)
  option(OPENCV_WITH_MSMF           "OpenCV: Build VideoIO with Media Foundation support"                                 OFF)
  option(OPENCV_WITH_XIMEA          "OpenCV: Include XIMEA cameras support"                                               OFF)
  option(OPENCV_WITH_XINE           "OpenCV: Include Xine support (GPL)"                                                  OFF)
  option(OPENCV_WITH_OPENCL         "OpenCV: Include OpenCL Runtime support"                                              OFF)
  option(OPENCV_WITH_OPENCLAMDFFT   "OpenCV: Include AMD OpenCL FFT library support"                                      OFF)
  option(OPENCV_WITH_OPENCLAMDBLAS  "OpenCV: Include AMD OpenCL BLAS library support"                                     OFF)
  option(OPENCV_WITH_INTELPERC      "OpenCV: Include Intel Perceptual Computing support"                                  OFF)

  # OpenCV build components
  option(OPENCV_BUILD_SHARED_LIBS      "OpenCV: Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
  option(OPENCV_BUILD_opencv_apps      "OpenCV: Build utility applications (used for example to train classifiers)" OFF)
  option(OPENCV_BUILD_ANDROID_EXAMPLES "OpenCV: Build examples for Android platform"                                OFF)
  option(OPENCV_BUILD_DOCS             "OpenCV: Create build rules for OpenCV Documentation"                        OFF)
  option(OPENCV_BUILD_EXAMPLES         "OpenCV: Build all examples"                                                 OFF)
  option(OPENCV_BUILD_PACKAGE          "OpenCV: Create build rules for OpenCV Documentation"                        OFF)
  option(OPENCV_BUILD_PERF_TESTS       "OpenCV: Build performance tests"                                            OFF)
  option(OPENCV_BUILD_TESTS            "OpenCV: Build accuracy & regression tests"                                  OFF)
  option(OPENCV_BUILD_WITH_DEBUG_INFO  "OpenCV: Include debug info into debug libs (not MSCV only)"                 OFF)
  option(OPENCV_BUILD_WITH_STATIC_CRT  "OpenCV: Enables use of staticaly linked CRT for staticaly linked OpenCV"    OFF)
  option(OPENCV_BUILD_FAT_JAVA_LIB     "OpenCV: Create fat java wrapper containing the whole OpenCV library"        OFF)
  option(OPENCV_BUILD_ANDROID_SERVICE  "OpenCV: Build OpenCV Manager for Google Play"                               OFF)
  option(OPENCV_BUILD_ANDROID_PACKAGE  "OpenCV: Build platform-specific package for Google Play"                    OFF)
  option(OPENCV_BUILD_TINY_GPU_MODULE  "OpenCV: Build tiny gpu module with limited image format support"            OFF)
  option(OPENCV_BUILD_ZLIB             "OpenCV: Build zlib from source"                                             ON)
  option(OPENCV_BUILD_TIFF             "OpenCV: Build libtiff from source"                                          ON)
  option(OPENCV_BUILD_JASPER           "OpenCV: Build libjasper from source"                                        OFF)
  if(WITH_LIBJPEG_TURBO)
    option(OPENCV_BUILD_JPEG           "OpenCV: Build libjpeg from source"                                          OFF)
  else()
    option(OPENCV_BUILD_JPEG           "OpenCV: Build libjpeg from source"                                          ON)
  endif()
  option(OPENCV_BUILD_PNG              "OpenCV: Build libpng from source"                                           ON)
  option(OPENCV_BUILD_OPENEXR          "OpenCV: Build openexr from source"                                          OFF)
  option(OPENCV_BUILD_TBB              "OpenCV: Download and build TBB from source"                                 OFF)

  # OpenCV installation options
  option(OPENCV_INSTALL_CREATE_DISTRIB   "OpenCV: Change install rules to build the distribution package"              OFF)
  option(OPENCV_INSTALL_C_EXAMPLES       "OpenCV: Install C examples"                                                  OFF)
  option(OPENCV_INSTALL_PYTHON_EXAMPLES  "OpenCV: Install Python examples"                                             OFF)
  option(OPENCV_INSTALL_ANDROID_EXAMPLES "OpenCV: Install Android examples"                                            OFF)
  option(OPENCV_INSTALL_TO_MANGLED_PATHS "OpenCV: Enables mangled install paths, that help with side by side installs" OFF)
  option(OPENCV_INSTALL_TESTS            "OpenCV: Install accuracy and performance test binaries and test data"        OFF)

  # OpenCV build options
  option(OPENCV_ENABLE_DYNAMIC_CUDA        "OpenCV: Enabled dynamic CUDA linkage"                             OFF)
  option(OPENCV_ENABLE_PRECOMPILED_HEADERS "OpenCV: Use precompiled headers"                                  OFF)
  option(OPENCV_ENABLE_SOLUTION_FOLDERS    "OpenCV: Solution folder in Visual Studio or in other IDEs"        OFF)
  option(OPENCV_ENABLE_PROFILING           "OpenCV: Enable profiling in the GCC compiler (Add flags: -g -pg)" OFF)
  option(OPENCV_ENABLE_COVERAGE            "OpenCV: Enable coverage collection with  GCov"                    OFF)
  option(OPENCV_ENABLE_OMIT_FRAME_POINTER  "OpenCV: Enable -fomit-frame-pointer for GCC"                      OFF)
  option(OPENCV_ENABLE_POWERPC             "OpenCV: Enable PowerPC for GCC"                                   OFF)
  option(OPENCV_ENABLE_FAST_MATH           "OpenCV: Enable -ffast-math (not recommended for GCC 4.6.x)"       OFF)
  option(OPENCV_ENABLE_SSE                 "OpenCV: Enable SSE instructions"                                  OFF)
  option(OPENCV_ENABLE_SSE2                "OpenCV: Enable SSE2 instructions"                                 OFF)
  option(OPENCV_ENABLE_SSE3                "OpenCV: Enable SSE3 instructions"                                 OFF)
  option(OPENCV_ENABLE_SSSE3               "OpenCV: Enable SSSE3 instructions"                                OFF)
  option(OPENCV_ENABLE_SSE41               "OpenCV: Enable SSE4.1 instructions"                               OFF)
  option(OPENCV_ENABLE_SSE42               "OpenCV: Enable SSE4.2 instructions"                               OFF)
  option(OPENCV_ENABLE_AVX                 "OpenCV: Enable AVX instructions"                                  OFF)
  option(OPENCV_ENABLE_AVX2                "OpenCV: Enable AVX2 instructions"                                 OFF)
  option(OPENCV_ENABLE_NEON                "OpenCV: Enable NEON instructions"                                 OFF)
  option(OPENCV_ENABLE_VFPV3               "OpenCV: Enable VFPv3-D32 instructions"                            OFF)
  option(OPENCV_ENABLE_NOISY_WARNINGS      "OpenCV: Show all warnings even if they are too noisy"             OFF)
  option(OPENCV_WARNINGS_ARE_ERRORS        "OpenCV: Treat warnings as errors"                                 OFF)
  option(OPENCV_ENABLE_WINRT_MODE          "OpenCV: Build with Windows Runtime support"                       OFF)
  option(OPENCV_ENABLE_WINRT_MODE_NATIVE   "OpenCV: Build with Windows Runtime native C++ support"            OFF)
  option(OPENCV_ENABLE_LIBVS2013           "OpenCV: Build VS2013 with Visual Studio 2013 libraries"           OFF)
  option(OPENCV_ENABLE_WINSDK81            "OpenCV: Build VS2013 with Windows 8.1 SDK"                        OFF)
  option(OPENCV_ENABLE_WINPHONESDK80       "OpenCV: Build with Windows Phone 8.0 SDK"                         OFF)
  option(OPENCV_ENABLE_WINPHONESDK81       "OpenCV: Build VS2013 with Windows Phone 8.1 SDK"                  OFF)

  # ============================

  # Download and build location
  set(OPENCV_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/opencv/source)
  set(OPENCV_BINARY_DIR ${PROJECT_BINARY_DIR}/download/opencv/build)

  # Get OpenCV from Git repository and build
  ExternalProject_Add(project_OpenCV
    PREFIX          ${CMAKE_INSTALL_PREFIX}
    TMP_DIR         ${OPENCV_BINARY_DIR}/tmp
    STAMP_DIR       ${OPENCV_BINARY_DIR}/stamp
    GIT_REPOSITORY  ${OPENCV_URL}
    GIT_TAG         ${OPENCV_TAG}
    SOURCE_DIR      ${OPENCV_SOURCE_DIR}
    BINARY_DIR      ${OPENCV_BINARY_DIR}
    BUILD_COMMAND   ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}
    INSTALL_DIR     ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}
    CMAKE_ARGS
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -D CMAKE_VERBOSE=${VERBOSE}
      -D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE}
      -D CMAKE_BUILD_TYPE=${OPENCV_BUILD_TYPE}
      -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
      -D CMAKE_SKIP_BUILD_RPATH=${CMAKE_SKIP_BUILD_RPATH}
      -D CMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=${CMAKE_INSTALL_RPATH_USE_LINK_PATH}
      -D CMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
      -D CMAKE_MACOSX_RPATH=${CMAKE_MACOSX_RPATH}
      -D BUILD_opencv_calib3d=${OPENCV_BUILD_opencv_calib3d}
      -D BUILD_opencv_contrib=${OPENCV_BUILD_opencv_contrib}
      -D BUILD_opencv_core=${OPENCV_BUILD_opencv_core}
      -D BUILD_opencv_dynamicuda=${OPENCV_BUILD_opencv_dynamicuda}
      -D BUILD_opencv_features2d=${OPENCV_BUILD_opencv_features2d}
      -D BUILD_opencv_flann=${OPENCV_BUILD_opencv_flann}
      -D BUILD_opencv_gpu=${OPENCV_BUILD_opencv_gpu}
      -D BUILD_opencv_highgui=${OPENCV_BUILD_opencv_highgui}
      -D BUILD_opencv_imgproc=${OPENCV_BUILD_opencv_imgproc}
      -D BUILD_opencv_java=${OPENCV_BUILD_opencv_java}
      -D BUILD_opencv_legacy=${OPENCV_BUILD_opencv_legacy}
      -D BUILD_opencv_ml=${OPENCV_BUILD_opencv_ml}
      -D BUILD_opencv_nonfree=${OPENCV_BUILD_opencv_nonfree}
      -D BUILD_opencv_objdetect=${OPENCV_BUILD_opencv_objdetect}
      -D BUILD_opencv_ocl=${OPENCV_BUILD_ocl_objdetect}
      -D BUILD_opencv_photo=${OPENCV_BUILD_opencv_photo}
      -D BUILD_opencv_python=${OPENCV_BUILD_opencv_python}
      -D BUILD_opencv_stitching=${OPENCV_BUILD_opencv_stitching}
      -D BUILD_opencv_superres=${OPENCV_BUILD_opencv_superres}
      -D BUILD_opencv_ts=${OPENCV_BUILD_opencv_ts}
      -D BUILD_opencv_video=${OPENCV_BUILD_opencv_video}
      -D BUILD_opencv_videostab=${OPENCV_BUILD_opencv_videostab}
      -D BUILD_opencv_viz=${OPENCV_BUILD_opencv_viz}
      -D BUILD_opencv_world=${OPENCV_BUILD_opencv_world}
      -D WITH_1394=${OPENCV_WITH_1394}
      -D WITH_AVFOUNDATION=${OPENCV_WITH_AVFOUNDATION}
      -D WITH_CARBON=${OPENCV_WITH_CARBON}
      -D WITH_VTK=${OPENCV_WITH_VTK}
      -D WITH_CUDA=${OPENCV_WITH_CUDA}
      -D WITH_CUFFT=${OPENCV_WITH_CUFFT}
      -D WITH_CUBLAS=${OPENCV_WITH_CUBLAS}
      -D WITH_NVCUVID=${OPENCV_WITH_NVCUVID}
      -D WITH_EIGEN=${OPENCV_WITH_EIGEN}
      -D WITH_VFW=${OPENCV_WITH_VFW}
      -D WITH_FFMPEG=${OPENCV_WITH_FFMPEG}
      -D WITH_GSTREAMER=${OPENCV_WITH_GSTREAMER}
      -D WITH_GSTREAMER_0_10=${OPENCV_WITH_GSTREAMER_0_10}
      -D WITH_GTK=${OPENCV_WITH_GTK}
      -D WITH_IPP=${OPENCV_WITH_IPP}
      -D WITH_JASPER=${OPENCV_WITH_JASPER}
      -D WITH_JPEG=${OPENCV_WITH_JPEG}
      -D WITH_OPENEXR=${OPENCV_WITH_OPENEXR}
      -D WITH_OPENGL=${OPENCV_WITH_OPENGL}
      -D WITH_OPENNI=${OPENCV_WITH_OPENNI}
      -D WITH_PNG=${OPENCV_WITH_PNG}
      -D WITH_PVAPI=${OPENCV_WITH_PVAPI}
      -D WITH_GIGEAPI=${OPENCV_WITH_GIGEAPI}
      -D WITH_QT=${OPENCV_WITH_QT}
      -D WITH_WIN32UI=${OPENCV_WITH_WIN32UI}
      -D WITH_QUICKTIME=${OPENCV_WITH_QUICKTIME}
      -D WITH_TBB=${OPENCV_WITH_TBB}
      -D WITH_OPENMP=${OPENCV_WITH_OPENMP}
      -D WITH_CSTRIPES=${OPENCV_WITH_CSTRIPES}
      -D WITH_TIFF=${OPENCV_WITH_TIFF}
      -D WITH_UNICAP=${OPENCV_WITH_UNICAP}
      -D WITH_V4L=${OPENCV_WITH_V4L}
      -D WITH_LIBV4L=${OPENCV_WITH_LIBV4L}
      -D WITH_DSHOW=${OPENCV_WITH_DSHOW}
      -D WITH_MSMF=${OPENCV_WITH_MSMF}
      -D WITH_XIMEA=${OPENCV_WITH_XIMEA}
      -D WITH_XINE=${OPENCV_WITH_XINE}
      -D WITH_OPENCL=${OPENCV_WITH_OPENCL}
      -D WITH_OPENCLAMDFFT=${OPENCV_WITH_OPENCLAMDFFT}
      -D WITH_OPENCLAMDBLAS=${OPENCV_WITH_OPENCLAMDBLAS}
      -D WITH_INTELPERC=${OPENCV_WITH_INTELPERC}
      -D BUILD_SHARED_LIBS=${OPENCV_BUILD_SHARED_LIBS}
      -D BUILD_opencv_apps=${OPENCV_BUILD_opencv_apps}
      -D BUILD_ANDROID_EXAMPLES=${OPENCV_BUILD_ANDROID_EXAMPLES}
      -D BUILD_DOCS=${OPENCV_BUILD_DOCS}
      -D BUILD_EXAMPLES=${OPENCV_BUILD_EXAMPLES}
      -D BUILD_PACKAGE=${OPENCV_BUILD_PACKAGE}
      -D BUILD_PERF_TESTS=${OPENCV_BUILD_PERF_TESTS}
      -D BUILD_TESTS=${OPENCV_BUILD_TESTS}
      -D BUILD_WITH_DEBUG_INFO=${OPENCV_BUILD_WITH_DEBUG_INFO}
      -D BUILD_WITH_STATIC_CRT=${OPENCV_BUILD_WITH_STATIC_CRT}
      -D BUILD_FAT_JAVA_LIB=${OPENCV_BUILD_FAT_JAVA_LIB}
      -D BUILD_ANDROID_SERVICE=${OPENCV_BUILD_ANDROID_SERVICE}
      -D BUILD_ZLIB=${OPENCV_BUILD_ZLIB}
      -D BUILD_TIFF=${OPENCV_BUILD_TIFF}
      -D BUILD_JASPER=${OPENCV_BUILD_JASPER}
      -D BUILD_JPEG=${OPENCV_BUILD_JPEG}
      -D BUILD_PNG=${OPENCV_BUILD_PNG}
      -D BUILD_OPENEXR=${OPENCV_BUILD_OPENEXR}
      -D BUILD_TBB=${OPENCV_BUILD_TBB}
      -D INSTALL_CREATE_DISTRIB=${OPENCV_INSTALL_CREATE_DISTRIB}
      -D INSTALL_C_EXAMPLES=${OPENCV_INSTALL_C_EXAMPLES}
      -D INSTALL_PYTHON_EXAMPLES=${OPENCV_INSTALL_PYTHON_EXAMPLES}
      -D INSTALL_ANDROID_EXAMPLES=${OPENCV_INSTALL_ANDROID_EXAMPLES}
      -D INSTALL_TO_MANGLED_PATHS=${OPENCV_INSTALL_TO_MANGLED_PATHS}
      -D INSTALL_TESTS=${OPENCV_INSTALL_TESTS}
      -D ENABLE_PRECOMPILED_HEADERS=${OPENCV_ENABLE_PRECOMPILED_HEADERS}
      -D ENABLE_SOLUTION_FOLDERS=${OPENCV_ENABLE_SOLUTION_FOLDERS}
      -D ENABLE_PROFILING=${OPENCV_ENABLE_PROFILING}
      -D ENABLE_COVERAGE=${OPENCV_ENABLE_COVERAGE}
      -D ENABLE_OMIT_FRAME_POINTER=${OPENCV_ENABLE_OMIT_FRAME_POINTER}
      -D ENABLE_POWERPC=${OPENCV_ENABLE_POWERPC}
      -D ENABLE_FAST_MATH=${OPENCV_ENABLE_FAST_MATH}
      -D ENABLE_SSE=${OPENCV_ENABLE_SSE}
      -D ENABLE_SSE2=${OPENCV_ENABLE_SSE2}
      -D ENABLE_SSE3=${OPENCV_ENABLE_SSE3}
      -D ENABLE_SSSE3=${OPENCV_ENABLE_SSSE3}
      -D ENABLE_SSE41=${OPENCV_ENABLE_SSE41}
      -D ENABLE_SSE42=${OPENCV_ENABLE_SSE42}
      -D ENABLE_AVX=${OPENCV_ENABLE_AVX}
      -D ENABLE_AVX2=${OPENCV_ENABLE_AVX2}
      -D ENABLE_NEON=${OPENCV_ENABLE_NEON}
      -D ENABLE_VFPV3=${OPENCV_ENABLE_VFPV3}
      -D ENABLE_NOISY_WARNINGS=${OPENCV_ENABLE_NOISY_WARNINGS}
      -D OPENCV_WARNINGS_ARE_ERRORS=${OPENCV_WARNINGS_ARE_ERRORS}
      -D WITH_IMAGEIO=${OPENCV_WITH_IMAGEIO}
      -D BUILD_ANDROID_PACKAGE=${OPENCV_BUILD_ANDROID_PACKAGE}
      -D BUILD_TINY_GPU_MODULE=${OPENCV_BUILD_TINY_GPU_MODULE}
      -D ENABLE_DYNAMIC_CUDA=${OPENCV_ENABLE_DYNAMIC_CUDA}
      -D ENABLE_WINRT_MODE=${OPENCV_ENABLE_WINRT_MODE}
      -D ENABLE_WINRT_MODE_NATIVE=${OPENCV_ENABLE_WINRT_MODE_NATIVE}
      -D ENABLE_LIBVS2013=${OPENCV_ENABLE_LIBVS2013}
      -D ENABLE_WINSDK81=${OPENCV_ENABLE_WINSDK81}
      -D ENABLE_WINPHONESDK80=${OPENCV_ENABLE_WINPHONESDK80}
      -D ENABLE_WINPHONESDK81=${OPENCV_ENABLE_WINPHONESDK81}
      -D JPEG_INCLUDE_DIR=${OPENCV_JPEG_INCLUDE_DIR}
      -D JPEG_LIBRARY=${OPENCV_JPEG_LIBRARY}
      ${OPENCV_LIBJPEG_TURBO_INC}
      ${OPENCV_LIBJPEG_TURBO_LIB}
  )

  if(BUILD_JPEG_TURBO)
    message(STATUS "setting up the dependency of OpenCV on jpeg-turbo")
    add_dependencies(project_OpenCV project_jpeg_turbo)
  endif()

  # Get install directory
  set(OpenCV_DIR ${CMAKE_INSTALL_PREFIX})

  # Get header files
  set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)

  # Get libraries
  if(OPENCV_BUILD_SHARED_LIBS)
    set(OpenCV_LIBRARIES
      ${OpenCV_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}opencv_core${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${OpenCV_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}opencv_imgproc${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${OpenCV_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}opencv_highgui${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
  else()
    set(OpenCV_LIBRARIES
      ${OpenCV_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}opencv_core${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${OpenCV_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}opencv_imgproc${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${OpenCV_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}opencv_highgui${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
  endif()

  # LBANN has built OpenCV
  set(LBANN_BUILT_OPENCV TRUE)

endif()

# Include header files
include_directories(${OpenCV_INCLUDE_DIRS})

# Add preprocessor flag for OpenCV
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_OPENCV")

# LBANN has access to OpenCV
set(LBANN_HAS_OPENCV TRUE)
