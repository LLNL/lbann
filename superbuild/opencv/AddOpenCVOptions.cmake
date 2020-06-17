# OpenCV modules
option(OPENCV_BUILD_opencv_core "OpenCV: Enable core module" ON)
option(OPENCV_BUILD_opencv_flann "OpenCV: Enable flann module" OFF)
option(OPENCV_BUILD_opencv_imgproc "OpenCV: Enable imgproc module" ON)
option(OPENCV_BUILD_opencv_highgui "OpenCV: Enable highgui module" OFF)
option(OPENCV_BUILD_opencv_features2d "OpenCV: Enable features2d module" OFF)
option(OPENCV_BUILD_opencv_calib3d "OpenCV: Enable calib3d module" OFF)
option(OPENCV_BUILD_opencv_ml "OpenCV: Enable ml module" OFF)
option(OPENCV_BUILD_opencv_video "OpenCV: Enable video module" OFF)
option(OPENCV_BUILD_opencv_legacy "OpenCV: Enable legacy module" OFF)
option(OPENCV_BUILD_opencv_objdetect "OpenCV: Enable objdetect module" OFF)
option(OPENCV_BUILD_opencv_photo "OpenCV: Enable photo module" OFF)
option(OPENCV_BUILD_opencv_gpu "OpenCV: Enable gpu module" OFF)
option(OPENCV_BUILD_opencv_nonfree "OpenCV: Enable nonfree module" OFF)
option(OPENCV_BUILD_opencv_contrib "OpenCV: Enable contrib module" OFF)
option(OPENCV_BUILD_opencv_java "OpenCV: Enable java module" OFF)
option(OPENCV_BUILD_opencv_python "OpenCV: Enable python module" OFF)
option(OPENCV_BUILD_opencv_python2 "OpenCV: Enable python2 module" OFF)
option(OPENCV_BUILD_opencv_python3 "OpenCV: Enable python3 module" OFF)
option(OPENCV_BUILD_opencv_stitching "OpenCV: Enable stitching module" OFF)
option(OPENCV_BUILD_opencv_superres "OpenCV: Enable superres module" OFF)
option(OPENCV_BUILD_opencv_ts "OpenCV: Enable ts module" OFF)
option(OPENCV_BUILD_opencv_videostab "OpenCV: Enable videostab module" OFF)
option(OPENCV_BUILD_opencv_world "OpenCV: Enable world module" OFF)
option(OPENCV_ENABLE_NONFREE "OpenCV: Enable non-free algorithms" OFF)
option(OPENCV_BUILD_opencv_videoio "OpenCV: Enable videoio module" OFF)
option(OPENCV_BUILD_opencv_dnn "OpenCV: Build dnn module and protobuf" OFF)

# Optional 3rd party components
option(OPENCV_WITH_1394 "OpenCV: Include IEEE1394 support" OFF)
option(OPENCV_WITH_AVFOUNDATION
  "OpenCV: Use AVFoundation for Video I/O (iOS/Mac)" OFF)
option(OPENCV_WITH_CARBON "OpenCV: Use Carbon for UI instead of Cocoa" OFF)
option(OPENCV_WITH_CAROTENE
  "OpenCV: Use NVidia carotene acceleration library for ARM platform" OFF)
option(OPENCV_WITH_CPUFEATURES "OpenCV: Use cpufeatures Android library" OFF)
option(OPENCV_WITH_VTK
  "OpenCV: Include VTK library support (and build opencv_viz module eiher)" OFF)
option(OPENCV_WITH_CUDA "OpenCV: Include NVidia Cuda Runtime support" OFF)
option(OPENCV_WITH_CUFFT
  "OpenCV: Include NVidia cuFFT library support" OFF)
option(OPENCV_WITH_CUBLAS "OpenCV: Include NVidia cuBLAS library support" OFF)
option(OPENCV_WITH_NVCUVID
  "OpenCV: Include NVidia Video Decoding library support" OFF)
option(OPENCV_WITH_EIGEN "OpenCV: Include Eigen2/Eigen3 support" OFF)
option(OPENCV_WITH_VFW "OpenCV: Include Video for Windows support" OFF)
option(OPENCV_WITH_FFMPEG "OpenCV: Include FFMPEG support" OFF)
option(OPENCV_WITH_GSTREAMER "OpenCV: Include Gstreamer support" OFF)
option(OPENCV_WITH_GSTREAMER_0_10
  "OpenCV: Enable Gstreamer 0.10 support (instead of 1.x)" OFF)
option(OPENCV_WITH_GTK "OpenCV: Include GTK support" OFF)
option(OPENCV_WITH_GTK_2_X "OpenCV: Use GTK version 2" OFF)
option(OPENCV_WITH_IMAGEIO "OpenCV: ImageIO support for OS X" OFF)
option(OPENCV_WITH_IPP "OpenCV: Include Intel IPP support" ON)
option(OPENCV_WITH_HALIDE "OpenCV: Include Halide support" OFF)
option(OPENCV_WITH_JASPER "OpenCV: Include JPEG2K support" OFF)
option(OPENCV_WITH_JPEG "OpenCV: Include JPEG support" ON)
option(OPENCV_WITH_WEBP "OpenCV: Include WebP support" OFF)
option(OPENCV_WITH_OPENEXR "OpenCV: Include ILM support via OpenEXR" OFF)
option(OPENCV_WITH_OPENGL "OpenCV: Include OpenGL support" OFF)
option(OPENCV_WITH_OPENVX "OpenCV: Include OpenVX support" OFF)
option(OPENCV_WITH_OPENNI "OpenCV: Include OpenNI support" OFF)
option(OPENCV_WITH_OPENNI2 "OpenCV: Include OpenNI2 support" OFF)
option(OPENCV_WITH_PNG "OpenCV: Include PNG support" ON)
option(OPENCV_WITH_GDCM "OpenCV: Include DICOM support" OFF)
option(OPENCV_WITH_PVAPI "OpenCV: Include Prosilica GigE support" OFF)
option(OPENCV_WITH_GIGEAPI "OpenCV: Include Smartek GigE support" OFF)
option(OPENCV_WITH_ARAVIS "OpenCV: Include Aravis GigE support" OFF)
option(OPENCV_WITH_QT "OpenCV: Build with Qt Backend support" OFF)
option(OPENCV_WITH_WIN32UI "OpenCV: Build with Win32 UI Backend support" OFF)
option(OPENCV_WITH_QUICKTIME "OpenCV: Use QuickTime for Video I/O" OFF)
option(OPENCV_WITH_QTKIT "OpenCV: Use QTKit Video I/O backend" OFF)
option(OPENCV_WITH_TBB "OpenCV: Include Intel TBB support" OFF)
option(OPENCV_WITH_OPENMP "OpenCV: Include OpenMP support" OFF)
option(OPENCV_WITH_CSTRIPES "OpenCV: Include C= support" OFF)
option(OPENCV_WITH_PTHREADS_PF "OpenCV: Use pthreads-based parallel_for" OFF)
option(OPENCV_WITH_TIFF "OpenCV: Include TIFF support" ON)
option(OPENCV_WITH_UNICAP "OpenCV: Include Unicap support (GPL)" OFF)
option(OPENCV_WITH_V4L "OpenCV: Include Video 4 Linux support" OFF)
option(OPENCV_WITH_LIBV4L "OpenCV: Use libv4l for Video 4 Linux support" OFF)
option(OPENCV_WITH_DSHOW "OpenCV: Build VideoIO with DirectShow support" OFF)
option(OPENCV_WITH_MSMF
  "OpenCV: Build VideoIO with Media Foundation support" OFF)
option(OPENCV_WITH_XIMEA "OpenCV: Include XIMEA cameras support" OFF)
option(OPENCV_WITH_XINE "OpenCV: Include Xine support (GPL)" OFF)
option(OPENCV_WITH_CLP "OpenCV: Include Clp support (EPL)" OFF)
option(OPENCV_WITH_OPENCL "OpenCV: Include OpenCL Runtime support" OFF)
option(OPENCV_WITH_OPENCL_SVM
  "OpenCV: Include OpenCL Shared Virtual Memory support" OFF)
option(OPENCV_WITH_OPENCLAMDFFT
  "OpenCV: Include AMD OpenCL FFT library support" OFF)
option(OPENCV_WITH_OPENCLAMDBLAS
  "OpenCV: Include AMD OpenCL BLAS library support" OFF)
option(OPENCV_WITH_DIRECTX "OpenCV: Include DirectX support" OFF)
option(OPENCV_WITH_INTELPERC
  "OpenCV: Include Intel Perceptual Computing support" OFF)
option(OPENCV_WITH_IPP_A "OpenCV: Include Intel IPP_A support" OFF)
option(OPENCV_WITH_MATLAB "OpenCV: Include Matlab support" OFF)
option(OPENCV_WITH_VA "OpenCV: Include VA support" OFF)
option(OPENCV_WITH_VA_INTEL "OpenCV: Include Intel VA-API/OpenCL support" OFF)
option(OPENCV_WITH_MFX "OpenCV: Include Intel Media SDK support" OFF)
option(OPENCV_WITH_GDAL "OpenCV: Include GDAL Support" OFF)
option(OPENCV_WITH_GPHOTO2 "OpenCV: Include gPhoto2 library support" OFF)
option(OPENCV_WITH_LAPACK "OpenCV: Include Lapack library support" OFF)
option(OPENCV_WITH_ITT "OpenCV: Include Intel ITT support" ON)

# OpenCV build components
option(OPENCV_BUILD_SHARED_LIBS
  "OpenCV: Build shared libraries instead of static ones" ON)
option(OPENCV_BUILD_opencv_apps
  "OpenCV: Build utility applications" OFF)
option(OPENCV_BUILD_ANDROID_EXAMPLES
  "OpenCV: Build examples for Android platform" OFF)
option(OPENCV_BUILD_DOCS
  "OpenCV: Create build rules for OpenCV Documentation" OFF)
option(OPENCV_BUILD_EXAMPLES "OpenCV: Build all examples" OFF)
option(OPENCV_BUILD_PACKAGE
  "OpenCV: Create build rules for OpenCV Documentation" OFF)
option(OPENCV_BUILD_PERF_TESTS "OpenCV: Build performance tests" OFF)
option(OPENCV_BUILD_TESTS "OpenCV: Build accuracy & regression tests" OFF)
option(OPENCV_BUILD_WITH_DEBUG_INFO
  "OpenCV: Include debug info into debug libs (not MSCV only)" OFF)
option(OPENCV_BUILD_WITH_STATIC_CRT
  "OpenCV: Enables use of staticaly linked CRT for staticaly linked OpenCV" OFF)
option(OPENCV_BUILD_WITH_DYNAMIC_IPP
  "OpenCV: Enables dynamic linking of IPP (only for standalone IPP)" OFF)
option(OPENCV_BUILD_FAT_JAVA_LIB
  "OpenCV: Create fat java wrapper containing the whole OpenCV library" OFF)
option(OPENCV_BUILD_ANDROID_SERVICE
  "OpenCV: Build OpenCV Manager for Google Play" OFF)
option(OPENCV_BUILD_ANDROID_PACKAGE
  "OpenCV: Build platform-specific package for Google Play" OFF)
option(OPENCV_BUILD_TINY_GPU_MODULE
  "OpenCV: Build tiny gpu module with limited image format support" OFF)
option(OPENCV_BUILD_CUDA_STUBS
  "OpenCV: Build CUDA modules stubs when no CUDA SDK" OFF)

# 3rd party libs
option(OPENCV_WITH_LIBJPEG_TURBO "OpenCV: Should OpenCV use libjpeg" OFF)

option(OPENCV_BUILD_ZLIB "OpenCV: Build zlib from source" ON)
option(OPENCV_BUILD_TIFF "OpenCV: Build libtiff from source" ON)
option(OPENCV_BUILD_JASPER "OpenCV: Build libjasper from source" OFF)
if (OPENCV_WITH_LIBJPEG_TURBO)
  option(OPENCV_BUILD_JPEG "OpenCV: Build libjpeg from source" OFF)
else()
  option(OPENCV_BUILD_JPEG "OpenCV: Build libjpeg from source" ON)
endif()
option(OPENCV_BUILD_PNG "OpenCV: Build libpng from source" ON)
option(OPENCV_BUILD_OPENEXR "OpenCV: Build openexr from source" OFF)
option(OPENCV_BUILD_TBB "OpenCV: Download and build TBB from source" OFF)
option(OPENCV_BUILD_IPP_IW "OpenCV: Build IPP IW from source" OFF)
option(OPENCV_BUILD_ITT "OpenCV: Build Intel ITT from source" ON)

# OpenCV installation options
option(OPENCV_INSTALL_CREATE_DISTRIB
  "OpenCV: Change install rules to build the distribution package" OFF)
option(OPENCV_INSTALL_C_EXAMPLES "OpenCV: Install C examples" OFF)
option(OPENCV_INSTALL_PYTHON_EXAMPLES "OpenCV: Install Python examples" OFF)
option(OPENCV_INSTALL_ANDROID_EXAMPLES "OpenCV: Install Android examples" OFF)
option(OPENCV_INSTALL_TO_MANGLED_PATHS
  "OpenCV: Enables mangled install paths" OFF)
option(OPENCV_INSTALL_TESTS
  "OpenCV: Install accuracy and performance test binaries and test data" OFF)

# OpenCV build options
option(OPENCV_ENABLE_CCACHE "OpenCV: Use ccache" OFF)
option(OPENCV_DYNAMIC_CUDA "OpenCV: Enabled dynamic CUDA linkage" OFF)
option(OPENCV_ENABLE_PRECOMPILED_HEADERS "OpenCV: Use precompiled headers" OFF)
option(OPENCV_ENABLE_SOLUTION_FOLDERS
  "OpenCV: Solution folder in Visual Studio or in other IDEs" OFF)
option(OPENCV_ENABLE_PROFILING
  "OpenCV: Enable profiling in the GCC compiler (Add flags: -g -pg)" ON)
option(OPENCV_ENABLE_COVERAGE
  "OpenCV: Enable coverage collection with  GCov" OFF)
option(OPENCV_ENABLE_OMIT_FRAME_POINTER
  "OpenCV: Enable -fomit-frame-pointer for GCC" ON)
option(OPENCV_ENABLE_POWERPC "OpenCV: Enable PowerPC for GCC" ON)
option(OPENCV_ENABLE_FAST_MATH
  "OpenCV: Enable -ffast-math (not recommended for GCC 4.6.x)" ON)
option(OPENCV_ENABLE_NEON "OpenCV: Enable NEON instructions" OFF)
option(OPENCV_ENABLE_VFPV3 "OpenCV: Enable VFPv3-D32 instructions" OFF)
option(OPENCV_ENABLE_NOISY_WARNINGS
  "OpenCV: Show all warnings even if they are too noisy" OFF)
option(OPENCV_WARNINGS_ARE_ERRORS "OpenCV: Treat warnings as errors" OFF)
option(OPENCV_ENABLE_WINRT_MODE
  "OpenCV: Build with Windows Runtime support" OFF)
option(OPENCV_ENABLE_WINRT_MODE_NATIVE
  "OpenCV: Build with Windows Runtime native C++ support" OFF)
option(OPENCV_ENABLE_LIBVS2013
  "OpenCV: Build VS2013 with Visual Studio 2013 libraries" OFF)
option(OPENCV_ENABLE_WINSDK81 "OpenCV: Build VS2013 with Windows 8.1 SDK" OFF)
option(OPENCV_ENABLE_WINPHONESDK80
  "OpenCV: Build with Windows Phone 8.0 SDK" OFF)
option(OPENCV_ENABLE_WINPHONESDK81
  "OpenCV: Build VS2013 with Windows Phone 8.1 SDK" OFF)
option(OPENCV_ANDROID_EXAMPLES_WITH_LIBS
  "OpenCV: Build binaries of Android examples with native libraries" OFF)
option(OPENCV_ENABLE_IMPL_COLLECTION
  "OpenCV: Collect implementation data on function call" OFF)
option(OPENCV_ENABLE_INSTRUMENTATION
  "OpenCV: Instrument functions to collect calls trace and performance" OFF)
option(OPENCV_ENABLE_GNU_STL_DEBUG
  "OpenCV: Enable GNU STL Debug mode (defines _GLIBCXX_DEBUG)" OFF)
option(OPENCV_ENABLE_BUILD_HARDENING
  "OpenCV: Enable hardening of the resulting binaries" OFF)
option(OPENCV_GENERATE_ABI_DESCRIPTOR
  "OpenCV: Generate XML file for abi_compliance_checker tool" OFF)
option(OPENCV_CV_ENABLE_INTRINSICS
  "OpenCV: Use intrinsic-based optimized code" ON)
option(OPENCV_CV_DISABLE_OPTIMIZATION
  "OpenCV: Disable explicit optimized code" OFF)
option(OPENCV_CV_TRACE "OpenCV: Enable OpenCV code trace" OFF)
