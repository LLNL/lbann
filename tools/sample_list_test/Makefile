MPICXX = mpic++

HDF5_DIR=/usr/workspace/wsb/icfsi/conduit-helpers/hdf5-quartz/hdf5
HDF5_INC=${HDF5_DIR}/include
HDF5_LIB=${HDF5_DIR}/lib
CONDUIT_DIR=/usr/workspace/wsb/icfsi/conduit/install-toss3-4.9.3-serial
CONDUIT_INC=${CONDUIT_DIR}/include
CONDUIT_LIB=${CONDUIT_DIR}/lib



all: sample_list.hpp sample_list_impl.hpp main.cpp
	${CXX} -std=c++11 -Wall -O3 -I${CONDUIT_INC} -I${HDF5_INC} -fopenmp -o test main.cpp -L${CONDUIT_LIB} -lconduit -lconduit_relay -L${HDF5_LIB} -lhdf5
	${MPICXX} -std=c++11 -Wall -lmpi -O3 -o test_mpi main_mpi.cpp

clean:
	rm -f *.o test test_mpi
