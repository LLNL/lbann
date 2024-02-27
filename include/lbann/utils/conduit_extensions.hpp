////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// http://github.com/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "conduit/conduit_data_array.hpp"
#include "conduit_relay_io_hdf5.hpp"
#include "hdf5.h"

namespace conduit {

template <typename T>
T
data_array_prod(DataArray<T> a)
{
    T res = 1;
    for(index_t i = 0; i < a.number_of_elements(); i++)
    {
        const T &val = a.element(i);
        res *= val;
    }

    return res;
}

namespace relay {
namespace io {
#define MAXDIMS 4
//-----------------------------------------------------------------------------
// This example tests reads of slabs from a hdf5 dataset.
//
// we may provide something like this in in the relay hdf5 interface
// in the future.
//-----------------------------------------------------------------------------
bool
hdf5_read_dset_slab(const std::string &file_path,
                    const std::string &fetch_path,
                    const DataType &dtype,
                    void *data_ptr)
{
    // assume fetch_path points to a hdf5 dataset
    // open the hdf5 file for reading
    hid_t h5_file_id = H5Fopen(file_path.c_str(),
                               H5F_ACC_RDONLY,
                               H5P_DEFAULT);
    CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
                     "Error opening HDF5 file for reading: "  << file_path);

    // open the dataset
    hid_t h5_dset_id = H5Dopen( h5_file_id, fetch_path.c_str(),H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_dset_id,
                     "Error opening HDF5 dataset at: " << fetch_path);


    // get info about the dataset
    hid_t h5_dspace_id = H5Dget_space(h5_dset_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_dspace_id,
                     "Error reading HDF5 Dataspace: " << h5_dset_id);

    // BVE

// int H5Sget_simple_extent_dims	(	hid_t 	space_id,
// hsize_t 	dims[],
// hsize_t 	maxdims[]
// )
    index_t rank = H5Sget_simple_extent_ndims(h5_dspace_id);

    std::cout << "The data space has a ndims of " << rank << std::endl;
    hsize_t dims[MAXDIMS] = {0,0,0,0};
    hsize_t maxdims[MAXDIMS] = {0,0,0,0};
    H5Sget_simple_extent_dims(h5_dspace_id, dims, maxdims);

    for(int i = 0; i < MAXDIMS; i++) {
      std::cout << "I think that I have dims["<<i<<"]=" << dims[i] <<std::endl;
      std::cout << "I think that I have maxdims["<<i<<"]=" << maxdims[i] <<std::endl;
    }
//    index_t rank       = H5Sget_simple_extent_ndims(dataspace_id);

    // check for empty case
    if(H5Sget_simple_extent_type(h5_dspace_id) == H5S_NULL)
    {
        // we have an error, but to try to clean up the hdf5 handles
        // before we issue the error.

        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_ERROR("Can't slab fetch from an empty hdf5 data set.");
    }

    hid_t h5_dtype_id  = H5Dget_type(h5_dset_id);

    CONDUIT_CHECK_HDF5_ERROR(h5_dtype_id,
                     "Error reading HDF5 Datatype: "
                     << h5_dset_id);

    // TODO: bounds check  (check that we are fetching a subset of the elems)
    index_t  h5_nelems = H5Sget_simple_extent_npoints(h5_dspace_id);
    if( dtype.number_of_elements() > h5_nelems)
    {
        // we have an error, but to try to clean up the hdf5 handles
        // before we issue the error.

        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_ERROR("Can't slab fetch a buffer larger than the source"
                        " hdf5 data set. Requested number of elements"
                        << dtype.number_of_elements()
                        << " hdf5 dataset number of elements" << h5_nelems);
    }


    // we need to compute an offset, stride, and element bytes
    // that will work for reading in the general case
    // right now we assume the dest type of data and the hdf5 datasets
    // data type are compatible

    // conduit's offsets, strides, are all in terms of bytes
    // hdf5's are in terms of elements

    // what we really want is a way to read bytes from the hdf5 dset with
    // out any type conversion, but that doesn't exist.

    // general support would include reading a a view of one type that
    //  points to a buffer of another
    // (for example a view of doubles that is defined on a buffer of bytes)

    // but hdf5 doesn't support slab fetch across datatypes
    // so for now we make sure the datatype is consistent.

    DataType h5_dt = conduit::relay::io::hdf5_dtype_to_conduit_dtype(h5_dtype_id,1);

    if( h5_dt.id() != dtype.id() )
    {
        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_INFO("Cannot fetch hdf5 slab of buffer and view are"
                       "different data types.")
        return false;
    }



    hid_t h5_status    = 0;

    hsize_t elem_bytes = dtype.element_bytes();
    hsize_t offset  = dtype.offset() / elem_bytes; // in bytes, convert to elems
    hsize_t stride  = dtype.stride() / elem_bytes; // in bytes, convert to elems
    hsize_t num_ele = dtype.number_of_elements();

    CONDUIT_INFO("slab dtype: " << dtype.to_json());

    CONDUIT_INFO("hdf5 slab: "  <<
                   " element_offset: " << offset <<
                   " element_stride: " << stride <<
                   " number_of_elements: " << num_ele);

    h5_status = H5Sselect_hyperslab(h5_dspace_id,
                                    H5S_SELECT_SET,
                                    &offset,
                                    &stride,
                                    &num_ele,
                                    0); // 0 here means NULL pointers; HDF5 *knows* dimension is 1
    // check subset sel
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                      "Error selecting hyper slab from HDF5 dataspace: " << h5_dspace_id);


    hid_t h5_dspace_compact_id = H5Screate_simple(1,
                                                  &num_ele,
                                                  NULL);

    CONDUIT_CHECK_HDF5_ERROR(h5_dspace_id,
                             "Failed to create HDF5 data space (memory dspace)");

    h5_status = H5Dread(h5_dset_id, // data set id
                        h5_dtype_id, // memory type id  // use same data type?
                        h5_dspace_compact_id,  // memory space id ...
                        h5_dspace_id, // file space id
                        H5P_DEFAULT,
                        data_ptr);
    // check read
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                      "Error reading bytes from HDF5 dataset: " << h5_dset_id);

    // close the data space
    CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                      "Error closing HDF5 data space: " << file_path);

    // close the compact data space
    CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_compact_id),
                      "Error closing HDF5 data space (memory dspace)" << file_path);


    // close the dataset
    CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                      "Error closing HDF5 dataset: " << file_path);

    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                      "Error closing HDF5 file: " << file_path);

    return true;
}

bool
hdf5_read_dataspace(const hid_t h5_file_id, //const std::string &file_path,
                    const std::string &fetch_path,
                    const DataType &dtype,
                    const conduit::Node &read_opts)
{
    // assume fetch_path points to a hdf5 dataset
    // open the hdf5 file for reading
    // hid_t h5_file_id = H5Fopen(file_path.c_str(),
    //                            H5F_ACC_RDONLY,
    //                            H5P_DEFAULT);
    // CONDUIT_CHECK_HDF5_ERROR(h5_file_id,
    //                  "Error opening HDF5 file for reading: "  << file_path);

    const std::string &file_path = "foo";

    // open the dataset
    hid_t h5_dset_id = H5Dopen( h5_file_id, fetch_path.c_str(),H5P_DEFAULT);

    CONDUIT_CHECK_HDF5_ERROR(h5_dset_id,
                     "Error opening HDF5 dataset at: " << fetch_path);


    // get info about the dataset
    hid_t h5_dspace_id = H5Dget_space(h5_dset_id);
    CONDUIT_CHECK_HDF5_ERROR(h5_dspace_id,
                     "Error reading HDF5 Dataspace: " << h5_dset_id);

    // check for empty case
    if(H5Sget_simple_extent_type(h5_dspace_id) == H5S_NULL)
    {
        // we have an error, but to try to clean up the hdf5 handles
        // before we issue the error.

        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_ERROR("Can't slab fetch from an empty hdf5 data set.");
    }

    hid_t h5_dtype_id  = H5Dget_type(h5_dset_id);

    CONDUIT_CHECK_HDF5_ERROR(h5_dtype_id,
                     "Error reading HDF5 Datatype: "
                     << h5_dset_id);

    // TODO: bounds check  (check that we are fetching a subset of the elems)
    index_t  h5_nelems = H5Sget_simple_extent_npoints(h5_dspace_id);
    if( dtype.number_of_elements() > h5_nelems)
    {
        // we have an error, but to try to clean up the hdf5 handles
        // before we issue the error.

        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_ERROR("Can't slab fetch a buffer larger than the source"
                        " hdf5 data set. Requested number of elements"
                        << dtype.number_of_elements()
                        << " hdf5 dataset number of elements" << h5_nelems);
    }


    // we need to compute an offset, stride, and element bytes
    // that will work for reading in the general case
    // right now we assume the dest type of data and the hdf5 datasets
    // data type are compatible

    // conduit's offsets, strides, are all in terms of bytes
    // hdf5's are in terms of elements

    // what we really want is a way to read bytes from the hdf5 dset with
    // out any type conversion, but that doesn't exist.

    // general support would include reading a a view of one type that
    //  points to a buffer of another
    // (for example a view of doubles that is defined on a buffer of bytes)

    // but hdf5 doesn't support slab fetch across datatypes
    // so for now we make sure the datatype is consistent.

    DataType h5_dt = conduit::relay::io::hdf5_dtype_to_conduit_dtype(h5_dtype_id,1);

    if( h5_dt.id() != dtype.id() )
    {
        CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                          "Error closing HDF5 data space: " << file_path);

        CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                          "Error closing HDF5 dataset: " << file_path);
        // close the hdf5 file
        CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                         "Error closing HDF5 file: " << file_path);

        CONDUIT_INFO("Cannot fetch hdf5 slab of buffer and view are"
                       "different data types.")
        return false;
    }



    hid_t h5_status    = 0;

    hsize_t elem_bytes = dtype.element_bytes();
    hsize_t offset  = dtype.offset() / elem_bytes; // in bytes, convert to elems
    hsize_t stride  = dtype.stride() / elem_bytes; // in bytes, convert to elems
    hsize_t num_ele = dtype.number_of_elements();

    CONDUIT_INFO("slab dtype: " << dtype.to_json());

    CONDUIT_INFO("hdf5 slab: "  <<
                   " element_offset: " << offset <<
                   " element_stride: " << stride <<
                   " number_of_elements: " << num_ele);

    h5_status = H5Sselect_hyperslab(h5_dspace_id,
                                    H5S_SELECT_SET,
                                    &offset,
                                    &stride,
                                    &num_ele,
                                    0); // 0 here means NULL pointers; HDF5 *knows* dimension is 1
    // check subset sel
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                      "Error selecting hyper slab from HDF5 dataspace: " << h5_dspace_id);


    // check read
    CONDUIT_CHECK_HDF5_ERROR(h5_status,
                      "Error reading bytes from HDF5 dataset: " << h5_dset_id);

    // close the data space
    CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_id),
                      "Error closing HDF5 data space: " << file_path);

    // // close the compact data space
    // CONDUIT_CHECK_HDF5_ERROR(H5Sclose(h5_dspace_compact_id),
    //                   "Error closing HDF5 data space (memory dspace)" << file_path);


    // close the dataset
    CONDUIT_CHECK_HDF5_ERROR(H5Dclose(h5_dset_id),
                      "Error closing HDF5 dataset: " << file_path);

    // close the hdf5 file
    CONDUIT_CHECK_HDF5_ERROR(H5Fclose(h5_file_id),
                      "Error closing HDF5 file: " << file_path);

    return h5_status;
    //    return true;
}

} // io

} // relay

} // conduit
