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
    set(OPENCV_TAG "3.3.0")
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
  option(OPENCV_ENABLE_NONFREE          "OpenCV: Enable non-free algorithms"                                         OFF) # 3.3.0

  # Optional 3rd party components
  option(OPENCV_WITH_1394           "OpenCV: Include IEEE1394 support"                                                    OFF) # both
  option(OPENCV_WITH_AVFOUNDATION   "OpenCV: Use AVFoundation for Video I/O (iOS/Mac)"                                    OFF) # both
  option(OPENCV_WITH_CARBON         "OpenCV: Use Carbon for UI instead of Cocoa"                                          OFF) # both
  option(OPENCV_WITH_CAROTENE       "OpenCV: Use NVidia carotene acceleration library for ARM platform"                   OFF) # 3.3.0
  option(OPENCV_WITH_CPUFEATURES    "OpenCV: Use cpufeatures Android library"                                             OFF) # 3.3.0
  option(OPENCV_WITH_VTK            "OpenCV: Include VTK library support (and build opencv_viz module eiher)"             OFF) # both
  option(OPENCV_WITH_CUDA           "OpenCV: Include NVidia Cuda Runtime support"                                         OFF) # both
  option(OPENCV_WITH_CUFFT          "OpenCV: Include NVidia Cuda Fast Fourier Transform (FFT) library support"            OFF) # both
  option(OPENCV_WITH_CUBLAS         "OpenCV: Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support" OFF) # both
  option(OPENCV_WITH_NVCUVID        "OpenCV: Include NVidia Video Decoding library support"                               OFF) # both
  option(OPENCV_WITH_EIGEN          "OpenCV: Include Eigen2/Eigen3 support"                                               OFF) # both
  option(OPENCV_WITH_VFW            "OpenCV: Include Video for Windows support"                                           OFF) # both
  option(OPENCV_WITH_FFMPEG         "OpenCV: Include FFMPEG support"                                                      OFF) # both
  option(OPENCV_WITH_GSTREAMER      "OpenCV: Include Gstreamer support"                                                   OFF) # both
  option(OPENCV_WITH_GSTREAMER_0_10 "OpenCV: Enable Gstreamer 0.10 support (instead of 1.x)"                              OFF) # both
  option(OPENCV_WITH_GTK            "OpenCV: Include GTK support"                                                         OFF) # both
  option(OPENCV_WITH_GTK_2_X        "OpenCV: Use GTK version 2"                                                           OFF) # 3.3.0
  option(OPENCV_WITH_IPP            "OpenCV: Include Intel IPP support"                                                   OFF) # Causes a hash mismatch error when downloading # both
  option(OPENCV_WITH_HALIDE         "OpenCV: Include Halide support"                                                      OFF) # 3.3.0
  option(OPENCV_WITH_JASPER         "OpenCV: Include JPEG2K support"                                                      OFF) # both
  option(OPENCV_WITH_JPEG           "OpenCV: Include JPEG support"                                                        ON) # both
  option(OPENCV_WITH_WEBP           "OpenCV: Include WebP support"                                                        OFF) # 3.3.0
  option(OPENCV_WITH_OPENEXR        "OpenCV: Include ILM support via OpenEXR"                                             OFF) # both
  option(OPENCV_WITH_OPENGL         "OpenCV: Include OpenGL support"                                                      OFF) # both
  option(OPENCV_WITH_OPENVX         "OpenCV: Include OpenVX support"                                                      OFF) # 3.3.0
  option(OPENCV_WITH_OPENNI         "OpenCV: Include OpenNI support"                                                      OFF) # both
  option(OPENCV_WITH_OPENNI2        "OpenCV: Include OpenNI2 support"                                                     OFF) # 3.3.0
  option(OPENCV_WITH_PNG            "OpenCV: Include PNG support"                                                         ON) # both
  option(OPENCV_WITH_GDCM           "OpenCV: Include DICOM support"                                                       OFF) # 3.3.0
  option(OPENCV_WITH_PVAPI          "OpenCV: Include Prosilica GigE support"                                              OFF) # both
  option(OPENCV_WITH_GIGEAPI        "OpenCV: Include Smartek GigE support"                                                OFF) # both
  option(OPENCV_WITH_ARAVIS         "OpenCV: Include Aravis GigE support"                                                 OFF) # 3.3.0
  option(OPENCV_WITH_QT             "OpenCV: Build with Qt Backend support"                                               OFF) # both
  option(OPENCV_WITH_WIN32UI        "OpenCV: Build with Win32 UI Backend support"                                         OFF) # both
  option(OPENCV_WITH_QUICKTIME      "OpenCV: Use QuickTime for Video I/O"                                                 OFF) # both
  option(OPENCV_WITH_QTKIT          "OpenCV: Use QTKit Video I/O backend"                                                 OFF) # 3.3.0
  option(OPENCV_WITH_TBB            "OpenCV: Include Intel TBB support"                                                   OFF) # both
  option(OPENCV_WITH_OPENMP         "OpenCV: Include OpenMP support"                                                      OFF) # both
  option(OPENCV_WITH_CSTRIPES       "OpenCV: Include C= support"                                                          OFF) # both
  option(OPENCV_WITH_PTHREADS_PF    "OpenCV: Use pthreads-based parallel_for"                                             OFF) # 3.3.0
  option(OPENCV_WITH_TIFF           "OpenCV: Include TIFF support"                                                        ON) # both
  option(OPENCV_WITH_UNICAP         "OpenCV: Include Unicap support (GPL)"                                                OFF) # both
  option(OPENCV_WITH_V4L            "OpenCV: Include Video 4 Linux support"                                               OFF) # both
  option(OPENCV_WITH_LIBV4L         "OpenCV: Use libv4l for Video 4 Linux support"                                        OFF) # both
  option(OPENCV_WITH_DSHOW          "OpenCV: Build VideoIO with DirectShow support"                                       OFF) # both
  option(OPENCV_WITH_MSMF           "OpenCV: Build VideoIO with Media Foundation support"                                 OFF) # both
  option(OPENCV_WITH_XIMEA          "OpenCV: Include XIMEA cameras support"                                               OFF) # both
  option(OPENCV_WITH_XINE           "OpenCV: Include Xine support (GPL)"                                                  OFF) # both
  option(OPENCV_WITH_CLP            "OpenCV: Include Clp support (EPL)"                                                   OFF) # 3.3.0
  option(OPENCV_WITH_OPENCL         "OpenCV: Include OpenCL Runtime support"                                              OFF) # both
  option(OPENCV_WITH_OPENCL_SVM     "OpenCV: Include OpenCL Shared Virtual Memory support"                                OFF) # experimental 3.3.0
  option(OPENCV_WITH_OPENCLAMDFFT   "OpenCV: Include AMD OpenCL FFT library support"                                      OFF) # both
  option(OPENCV_WITH_OPENCLAMDBLAS  "OpenCV: Include AMD OpenCL BLAS library support"                                     OFF) # both
  option(OPENCV_WITH_DIRECTX        "OpenCV: Include DirectX support"                                                     OFF) # 3.3.0
  option(OPENCV_WITH_INTELPERC      "OpenCV: Include Intel Perceptual Computing support"                                  OFF) # both
  option(OPENCV_WITH_IPP_A          "OpenCV: Include Intel IPP_A support"                                                 OFF) # 3.3.0
  option(OPENCV_WITH_MATLAB         "OpenCV: Include Matlab support"                                                      OFF) # 3.3.0
  option(OPENCV_WITH_VA             "OpenCV: Include VA support"                                                          OFF) # 3.3.0
  option(OPENCV_WITH_VA_INTEL       "OpenCV: Include Intel VA-API/OpenCL support"                                         OFF) # 3.3.0
  option(OPENCV_WITH_MFX            "OpenCV: Include Intel Media SDK support"                                             OFF) # 3.3.0
  option(OPENCV_WITH_GDAL           "OpenCV: Include GDAL Support"                                                        OFF) # 3.3.0
  option(OPENCV_WITH_GPHOTO2        "OpenCV: Include gPhoto2 library support"                                             OFF) # 3.3.0
  option(OPENCV_WITH_LAPACK         "OpenCV: Include Lapack library support"                                              OFF) # 3.3.0
  option(OPENCV_WITH_ITT            "OpenCV: Include Intel ITT support"                                                   ON) # 3.3.0

  # OpenCV build components
  option(OPENCV_BUILD_SHARED_LIBS      "OpenCV: Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON) # both
  option(OPENCV_BUILD_opencv_apps      "OpenCV: Build utility applications (used for example to train classifiers)" OFF) # both
  option(OPENCV_BUILD_ANDROID_EXAMPLES "OpenCV: Build examples for Android platform"                                OFF) # both
  option(OPENCV_BUILD_DOCS             "OpenCV: Create build rules for OpenCV Documentation"                        OFF) # both
  option(OPENCV_BUILD_EXAMPLES         "OpenCV: Build all examples"                                                 OFF) # both
  option(OPENCV_BUILD_PACKAGE          "OpenCV: Create build rules for OpenCV Documentation"                        OFF) # both
  option(OPENCV_BUILD_PERF_TESTS       "OpenCV: Build performance tests"                                            OFF) # both
  option(OPENCV_BUILD_TESTS            "OpenCV: Build accuracy & regression tests"                                  OFF) # both
  option(OPENCV_BUILD_WITH_DEBUG_INFO  "OpenCV: Include debug info into debug libs (not MSCV only)"                 OFF) # both
  option(OPENCV_BUILD_WITH_STATIC_CRT  "OpenCV: Enables use of staticaly linked CRT for staticaly linked OpenCV"    OFF) # both
  option(OPENCV_BUILD_WITH_DYNAMIC_IPP "OpenCV: Enables dynamic linking of IPP (only for standalone IPP)"           OFF) # 3.3.0
  option(OPENCV_BUILD_FAT_JAVA_LIB     "OpenCV: Create fat java wrapper containing the whole OpenCV library"        OFF) # both
  option(OPENCV_BUILD_ANDROID_SERVICE  "OpenCV: Build OpenCV Manager for Google Play"                               OFF) # both
  option(OPENCV_BUILD_CUDA_STUBS       "OpenCV: Build CUDA modules stubs when no CUDA SDK"                          OFF) # 3.3.0

# 3rd party libs
  option(OPENCV_BUILD_ZLIB             "OpenCV: Build zlib from source"                                             ON) # both
  option(OPENCV_BUILD_TIFF             "OpenCV: Build libtiff from source"                                          ON) # both
  option(OPENCV_BUILD_JASPER           "OpenCV: Build libjasper from source"                                        OFF) # both
  if(WITH_LIBJPEG_TURBO)
    option(OPENCV_BUILD_JPEG           "OpenCV: Build libjpeg from source"                                          OFF) # both
  else()
    option(OPENCV_BUILD_JPEG           "OpenCV: Build libjpeg from source"                                          ON) # both
  endif()
  option(OPENCV_BUILD_PNG              "OpenCV: Build libpng from source"                                           ON) # both
  option(OPENCV_BUILD_OPENEXR          "OpenCV: Build openexr from source"                                          OFF) # both
  option(OPENCV_BUILD_TBB              "OpenCV: Download and build TBB from source"                                 OFF) # both
  option(OPENCV_BUILD_IPP_IW           "OpenCV: Build IPP IW from source"                                           ON) # 3.3.0
  option(OPENCV_BUILD_ITT              "OpenCV: Build Intel ITT from source"                                        ON) # 3.3.0

  # OpenCV installation options
  option(OPENCV_INSTALL_CREATE_DISTRIB   "OpenCV: Change install rules to build the distribution package"              OFF) # both
  option(OPENCV_INSTALL_C_EXAMPLES       "OpenCV: Install C examples"                                                  OFF) # both
  option(OPENCV_INSTALL_PYTHON_EXAMPLES  "OpenCV: Install Python examples"                                             OFF) # both
  option(OPENCV_INSTALL_ANDROID_EXAMPLES "OpenCV: Install Android examples"                                            OFF) # both
  option(OPENCV_INSTALL_TO_MANGLED_PATHS "OpenCV: Enables mangled install paths, that help with side by side installs" OFF) # both
  option(OPENCV_INSTALL_TESTS            "OpenCV: Install accuracy and performance test binaries and test data"        OFF) # both

  # OpenCV build options
  option(OPENCV_ENABLE_CCACHE              "OpenCV: Use ccache"                                               OFF) # 3.3.0
  option(OPENCV_ENABLE_PRECOMPILED_HEADERS "OpenCV: Use precompiled headers"                                  OFF) # both
  option(OPENCV_ENABLE_SOLUTION_FOLDERS    "OpenCV: Solution folder in Visual Studio or in other IDEs"        OFF) # both
  option(OPENCV_ENABLE_PROFILING           "OpenCV: Enable profiling in the GCC compiler (Add flags: -g -pg)" ON) # both
  option(OPENCV_ENABLE_COVERAGE            "OpenCV: Enable coverage collection with  GCov"                    OFF) # both
  option(OPENCV_ENABLE_OMIT_FRAME_POINTER  "OpenCV: Enable -fomit-frame-pointer for GCC"                      ON) # both
  option(OPENCV_ENABLE_POWERPC             "OpenCV: Enable PowerPC for GCC"                                   ON) # both
  option(OPENCV_ENABLE_FAST_MATH           "OpenCV: Enable -ffast-math (not recommended for GCC 4.6.x)"       ON) # both
  option(OPENCV_ENABLE_NEON                "OpenCV: Enable NEON instructions"                                 OFF) # both
  option(OPENCV_ENABLE_VFPV3               "OpenCV: Enable VFPv3-D32 instructions"                            OFF) # both
  option(OPENCV_ENABLE_NOISY_WARNINGS      "OpenCV: Show all warnings even if they are too noisy"             OFF) # both
  option(OPENCV_WARNINGS_ARE_ERRORS        "OpenCV: Treat warnings as errors"                                 OFF) # both
  option(OPENCV_ANDROID_EXAMPLES_WITH_LIBS "OpenCV: Build binaries of Android examples with native libraries" OFF) # 3.3.0
  option(OPENCV_ENABLE_IMPL_COLLECTION     "OpenCV: Collect implementation data on function call"             OFF) # 3.3.0
  option(OPENCV_ENABLE_INSTRUMENTATION     "OpenCV: Instrument functions to collect calls trace and performance" OFF) # 3.3.0
  option(OPENCV_ENABLE_GNU_STL_DEBUG       "OpenCV: Enable GNU STL Debug mode (defines _GLIBCXX_DEBUG)"       OFF) # 3.3.0
  option(OPENCV_ENABLE_BUILD_HARDENING     "OpenCV: Enable hardening of the resulting binaries (against security attacks, detects memory corruption, etc)" OFF) # 3.3.0
  option(OPENCV_GENERATE_ABI_DESCRIPTOR    "OpenCV: Generate XML file for abi_compliance_checker tool"        OFF) # 3.3.0
  option(OPENCV_CV_ENABLE_INTRINSICS       "OpenCV: Use intrinsic-based optimized code"                       ON) # 3.3.0
  option(OPENCV_CV_DISABLE_OPTIMIZATION    "OpenCV: Disable explicit optimized code (dispatched code/intrinsics/loop unrolling/etc)" OFF) # 3.3.0
  option(OPENCV_CV_TRACE                   "OpenCV: Enable OpenCV code trace"                                 OFF) # 3.3.0

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
      -D CMAKE_C_FLAGS=${CMAKE_C_FLAGS}
      -D CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -D CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}
      -D CMAKE_SKIP_BUILD_RPATH=${CMAKE_SKIP_BUILD_RPATH}
      -D CMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=${CMAKE_INSTALL_RPATH_USE_LINK_PATH}
      -D CMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
      -D CMAKE_MACOSX_RPATH=${CMAKE_MACOSX_RPATH}
      -D OPENCV_ENABLE_NONFREE=${OPENCV_ENABLE_NONFREE}
      -D WITH_1394=${OPENCV_WITH_1394}
      -D WITH_AVFOUNDATION=${OPENCV_WITH_AVFOUNDATION}
      -D WITH_CARBON=${OPENCV_WITH_CARBON}
      -D WITH_CAROTENE=${OPENCV_WITH_CAROTENE}
      -D WITH_CPUFEATURES=${OPENCV_WITH_CPUFEATURES}
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
      -D WITH_GTK_2_X=${OPENCV_WITH_GTK_2_X}
      -D WITH_IPP=${OPENCV_WITH_IPP}
      -D WITH_HALIDE=${OPENCV_WITH_HALIDE}
      -D WITH_JASPER=${OPENCV_WITH_JASPER}
      -D WITH_JPEG=${OPENCV_WITH_JPEG}
      -D WITH_WEBP=${OPENCV_WITH_WEBP}
      -D WITH_OPENEXR=${OPENCV_WITH_OPENEXR}
      -D WITH_OPENGL=${OPENCV_WITH_OPENGL}
      -D WITH_OPENVX=${OPENCV_WITH_OPENVX}
      -D WITH_OPENNI=${OPENCV_WITH_OPENNI}
      -D WITH_OPENNI2=${OPENCV_WITH_OPENNI2}
      -D WITH_PNG=${OPENCV_WITH_PNG}
      -D WITH_GDCM=${OPENCV_WITH_GDCM}
      -D WITH_PVAPI=${OPENCV_WITH_PVAPI}
      -D WITH_GIGEAPI=${OPENCV_WITH_GIGEAPI}
      -D WITH_ARAVIS=${OPENCV_WITH_ARAVIS}
      -D WITH_QT=${OPENCV_WITH_QT}
      -D WITH_WIN32UI=${OPENCV_WITH_WIN32UI}
      -D WITH_QUICKTIME=${OPENCV_WITH_QUICKTIME}
      -D WITH_QTKIT=${OPENCV_WITH_QTKIT}
      -D WITH_TBB=${OPENCV_WITH_TBB}
      -D WITH_OPENMP=${OPENCV_WITH_OPENMP}
      -D WITH_CSTRIPES=${OPENCV_WITH_CSTRIPES}
      -D WITH_PTHREADS_PF=${OPENCV_WITH_PTHREADS_PF}
      -D WITH_TIFF=${OPENCV_WITH_TIFF}
      -D WITH_UNICAP=${OPENCV_WITH_UNICAP}
      -D WITH_V4L=${OPENCV_WITH_V4L}
      -D WITH_LIBV4L=${OPENCV_WITH_LIBV4L}
      -D WITH_DSHOW=${OPENCV_WITH_DSHOW}
      -D WITH_MSMF=${OPENCV_WITH_MSMF}
      -D WITH_XIMEA=${OPENCV_WITH_XIMEA}
      -D WITH_XINE=${OPENCV_WITH_XINE}
      -D WITH_CLP=${OPENCV_WITH_CLP}
      -D WITH_OPENCL=${OPENCV_WITH_OPENCL}
      -D WITH_OPENCL_SVM=${OPENCV_WITH_OPENCL_SVM}
      -D WITH_OPENCLAMDFFT=${OPENCV_WITH_OPENCLAMDFFT}
      -D WITH_OPENCLAMDBLAS=${OPENCV_WITH_OPENCLAMDBLAS}
      -D WITH_DIRECTX=${OPENCV_WITH_DIRECTX}
      -D WITH_INTELPERC=${OPENCV_WITH_INTELPERC}
      -D WITH_IPP_A=${OPENCV_WITH_IPP_A}
      -D WITH_MATLAB=${OPENCV_WITH_MATLAB}
      -D WITH_VA=${OPENCV_WITH_VA}
      -D WITH_VA_INTEL=${OPENCV_WITH_VA_INTEL}
      -D WITH_MFX=${OPENCV_WITH_MFX}
      -D WITH_GDAL=${OPENCV_WITH_GDAL}
      -D WITH_GPHOTO2=${OPENCV_WITH_GPHOTO2}
      -D WITH_LAPACK=${OPENCV_WITH_LAPACK}
      -D WITH_ITT=${OPENCV_WITH_ITT}
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
      -D BUILD_WITH_DYNAMIC_IPP=${OPENCV_BUILD_WITH_DYNAMIC_IPP}
      -D BUILD_FAT_JAVA_LIB=${OPENCV_BUILD_FAT_JAVA_LIB}
      -D BUILD_ANDROID_SERVICE=${OPENCV_BUILD_ANDROID_SERVICE}
      -D BUILD_CUDA_STUBS=${OPENCV_BUILD_CUDA_STUBS}
      -D BUILD_ZLIB=${OPENCV_BUILD_ZLIB}
      -D BUILD_TIFF=${OPENCV_BUILD_TIFF}
      -D BUILD_JASPER=${OPENCV_BUILD_JASPER}
      -D BUILD_JPEG=${OPENCV_BUILD_JPEG}
      -D BUILD_PNG=${OPENCV_BUILD_PNG}
      -D BUILD_OPENEXR=${OPENCV_BUILD_OPENEXR}
      -D BUILD_TBB=${OPENCV_BUILD_TBB}
      -D BUILD_IPP_IW=${OPENCV_BUILD_IPP_IW}
      -D BUILD_ITT=${OPENCV_BUILD_ITT}
      -D INSTALL_CREATE_DISTRIB=${OPENCV_INSTALL_CREATE_DISTRIB}
      -D INSTALL_C_EXAMPLES=${OPENCV_INSTALL_C_EXAMPLES}
      -D INSTALL_PYTHON_EXAMPLES=${OPENCV_INSTALL_PYTHON_EXAMPLES}
      -D INSTALL_ANDROID_EXAMPLES=${OPENCV_INSTALL_ANDROID_EXAMPLES}
      -D INSTALL_TO_MANGLED_PATHS=${OPENCV_INSTALL_TO_MANGLED_PATHS}
      -D INSTALL_TESTS=${OPENCV_INSTALL_TESTS}
      -D ENABLE_CCACHE=${OPENCV_ENABLE_CCACHE}
      -D ENABLE_PRECOMPILED_HEADERS=${OPENCV_ENABLE_PRECOMPILED_HEADERS}
      -D ENABLE_SOLUTION_FOLDERS=${OPENCV_ENABLE_SOLUTION_FOLDERS}
      -D ENABLE_PROFILING=${OPENCV_ENABLE_PROFILING}
      -D ENABLE_COVERAGE=${OPENCV_ENABLE_COVERAGE}
      -D ENABLE_OMIT_FRAME_POINTER=${OPENCV_ENABLE_OMIT_FRAME_POINTER}
      -D ENABLE_POWERPC=${OPENCV_ENABLE_POWERPC}
      -D ENABLE_FAST_MATH=${OPENCV_ENABLE_FAST_MATH}
      -D ENABLE_NEON=${OPENCV_ENABLE_NEON}
      -D ENABLE_VFPV3=${OPENCV_ENABLE_VFPV3}
      -D ENABLE_NOISY_WARNINGS=${OPENCV_ENABLE_NOISY_WARNINGS}
      -D OPENCV_WARNINGS_ARE_ERRORS=${OPENCV_WARNINGS_ARE_ERRORS}
      -D ANDROID_EXAMPLES_WITH_LIBS=${OPENCV_ANDROID_EXAMPLES_WITH_LIBS}
      -D ENABLE_IMPL_COLLECTION=${OPENCV_ENABLE_IMPL_COLLECTION}
      -D ENABLE_INSTRUMENTATION=${OPENCV_ENABLE_INSTRUMENTATION}
      -D ENABLE_GNU_STL_DEBUG=${OPENCV_ENABLE_GNU_STL_DEBUG}
      -D ENABLE_BUILD_HARDENING=${OPENCV_ENABLE_BUILD_HARDENING}
      -D GENERATE_ABI_DESCRIPTOR=${OPENCV_GENERATE_ABI_DESCRIPTOR}
      -D CV_ENABLE_INTRINSICS=${OPENCV_CV_ENABLE_INTRINSICS}
      -D CV_DISABLE_OPTIMIZATION=${OPENCV_CV_DISABLE_OPTIMIZATION}
      -D CV_TRACE=${OPENCV_CV_TRACE}
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
add_definitions(-D__LIB_OPENCV)

# LBANN has access to OpenCV
set(LBANN_HAS_OPENCV TRUE)
