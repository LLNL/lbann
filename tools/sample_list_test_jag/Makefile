MPICXX = mpic++

HDF5_DIR=/usr/workspace/wsb/icfsi/conduit-helpers/hdf5-quartz/hdf5
HDF5_INC=${HDF5_DIR}/include
HDF5_LIB=${HDF5_DIR}/lib
CONDUIT_DIR=/usr/workspace/wsb/icfsi/conduit/install-toss3-4.9.3-serial
CONDUIT_INC=${CONDUIT_DIR}/include
CONDUIT_LIB=${CONDUIT_DIR}/lib
LBANN_DIR=../..


all: run_distributed run_centralized


run_distributed: sample_list_jag.hpp sample_list_jag_impl.hpp main_distributed.cpp file_utils.o
	${CXX} -std=c++11 -g -Wall -O3 -I. -I${CONDUIT_INC} -I${HDF5_INC} -fopenmp -o run_distributed main_distributed.cpp file_utils.o -L${CONDUIT_LIB} -lconduit -lconduit_relay -L${HDF5_LIB} -lhdf5

run_centralized: sample_list_jag.hpp sample_list_jag_impl.hpp main_centralized.cpp file_utils.o
	${MPICXX} -std=c++11 -g -Wall -O3 -I. -I${CONDUIT_INC} -I${HDF5_INC} -fopenmp -o run_centralized main_centralized.cpp file_utils.o -L${CONDUIT_LIB} -lconduit -lconduit_relay -L${HDF5_LIB} -lhdf5 -lmpi

file_utils.o: ${LBANN_DIR}/src/utils/file_utils.cpp
	${CXX} -std=c++11 -g -Wall -O3 -I. -c ${LBANN_DIR}/src/utils/file_utils.cpp


clean:
	rm -f *.o run_centralized run_distributed
