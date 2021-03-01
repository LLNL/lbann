# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  CIRC_CIRC.cpp
  MC_MR.cpp
  MC_STAR.cpp
  MD_STAR.cpp
  MR_MC.cpp
  MR_STAR.cpp
  Matrix.cpp
  STAR_MC.cpp
  STAR_MD.cpp
  STAR_MR.cpp
  STAR_STAR.cpp
  STAR_VC.cpp
  STAR_VR.cpp
  VC_STAR.cpp
  VR_STAR.cpp
  common.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
