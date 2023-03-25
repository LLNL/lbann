################################################################################
## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################
set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  center_crop_test.cpp
  colorize_test.cpp
  grayscale_test.cpp
  horizontal_flip_test.cpp
  random_affine_test.cpp
  random_crop_test.cpp
  random_resized_crop_test.cpp
  random_resized_crop_with_fixed_aspect_ratio_test.cpp
  resize_test.cpp
  resized_center_crop_test.cpp
  to_lbann_layout_test.cpp
  transform_pipeline_test.cpp
  vertical_flip_test.cpp
  )

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}"
  PARENT_SCOPE)
