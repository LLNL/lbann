# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  data_reader.cpp
  data_reader_cifar10.cpp
  data_reader_csv.cpp
  data_reader_image.cpp
  data_reader_jag_conduit.cpp
  data_reader_merge_features.cpp
  data_reader_merge_samples.cpp
  data_reader_mesh.cpp
  data_reader_mnist.cpp
  data_reader_nci.cpp
  data_reader_node2vec.cpp
  data_reader_synthetic.cpp
  data_reader_python.cpp
  data_reader_smiles.cpp
  data_reader_HDF5.cpp
  data_reader_sample_list.cpp
  )

if (LBANN_HAS_CNPY)
  set_full_path(CNPY_SRC_FILES
    data_reader_npz_ras_lipid.cpp
    data_reader_numpy.cpp
    data_reader_numpy_npz.cpp
    data_reader_numpy_npz_conduit.cpp
    data_reader_pilot2_molecular.cpp
    )
  list(APPEND THIS_DIR_SOURCES ${CNPY_SRC_FILES})
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/data_reader_hdf5_legacy.cpp")
endif ()

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/data_reader_imagenet.cpp")
endif ()

# Add the subdirectories
add_subdirectory(utils)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
