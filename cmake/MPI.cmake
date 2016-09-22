# Try finding MPI
find_package(MPI REQUIRED)

# Status message
message(STATUS "Found MPI: ${MPI_CXX_COMPILER} ${MPI_C_COMPILER} ${MPI_Fortran_COMPILER}")

# Include MPI header files
include_directories(${MPI_CXX_INCLUDE_PATH})

# LBANN has access to MPI
set(LBANN_HAS_MPI TRUE)
