MPICXX = mpic++

all: sample_list.hpp sample_list_impl.hpp main.cpp
	${CXX} -std=c++11 -Wall -o test main.cpp
	${MPICXX} -std=c++11 -Wall -lmpi -o test_mpi main_mpi.cpp

clean:
	rm -f *.o test test_mpi
