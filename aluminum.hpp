#ifndef HYDROGEN_IMPORTS_ALUMINUM_HPP_
#define HYDROGEN_IMPORTS_ALUMINUM_HPP_

#ifdef HYDROGEN_HAVE_ALUMINUM
#include <Al.hpp>
#endif // HYDROGEN_HAVE_ALUMINUM

namespace El
{

// FIXME: This is a lame shortcut to save some
// metaprogramming. Deadlines are the worst.
enum class Collective
{
    ALLGATHER,
    ALLREDUCE,
    ALLTOALL,
    BROADCAST,
    GATHER,
    REDUCE,
    REDUCESCATTER,
    SCATTER,
    SENDRECV
};// enum class Collective

#ifndef HYDROGEN_HAVE_ALUMINUM

template <typename T> struct IsAluminumTypeT : std::false_type {};
template <typename T, Device D>
struct IsAluminumDeviceType : std::false_type {};
template <typename T, Device D, Collective C>
struct IsAluminumSupported : std::false_type {};

#else

// A function to convert an MPI MPI_Op into an Aluminum operator
Al::ReductionOperator MPI_Op2ReductionOperator(MPI_Op op);

#define ADD_ALUMINUM_TYPE(type, backend) \
    template <> struct IsAlTypeT<type,backend> : std::true_type {}
#define ADD_ALUMINUM_COLLECTIVE(coll, backend) \
    template <> struct IsBackendSupported<coll, backend> : std::true_type {}

//
// Setup type support
//

template <typename T, typename BackendT>
struct IsAlTypeT : std::false_type {};

ADD_ALUMINUM_TYPE(              char, Al::MPIBackend);
ADD_ALUMINUM_TYPE(       signed char, Al::MPIBackend);
ADD_ALUMINUM_TYPE(     unsigned char, Al::MPIBackend);
ADD_ALUMINUM_TYPE(             short, Al::MPIBackend);
ADD_ALUMINUM_TYPE(    unsigned short, Al::MPIBackend);
ADD_ALUMINUM_TYPE(               int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(      unsigned int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(          long int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(     long long int, Al::MPIBackend);
ADD_ALUMINUM_TYPE( unsigned long int, Al::MPIBackend);
ADD_ALUMINUM_TYPE(unsigned long long, Al::MPIBackend);
ADD_ALUMINUM_TYPE(             float, Al::MPIBackend);
ADD_ALUMINUM_TYPE(            double, Al::MPIBackend);
ADD_ALUMINUM_TYPE(       long double, Al::MPIBackend);

#if 1
#ifdef HYDROGEN_HAVE_NCCL2
ADD_ALUMINUM_TYPE(                  char, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(         unsigned char, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(                   int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(          unsigned int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(         long long int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(unsigned long long int, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(                 float, Al::NCCLBackend);
ADD_ALUMINUM_TYPE(                double, Al::NCCLBackend);
#endif // HYDROGEN_HAVE_NCCL2
#endif
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
template <typename T>
struct IsAlTypeT<T, Al::MPICUDABackend> : IsAlTypeT<T, Al::MPIBackend> {};
#endif // HYDROGEN_HAVE_AL_MPI_CUDA

//
// Setup collective support
//

template <Collective C, typename BackendT>
struct IsBackendSupported : std::false_type {};

// MPI backend only supports AllReduce
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLREDUCE, Al::MPIBackend);

#if 1
#ifdef HYDROGEN_HAVE_NCCL2
// NCCL backend supports these
//ADD_ALUMINUM_COLLECTIVE(    Collective::ALLGATHER, Al::NCCLBackend);
//ADD_ALUMINUM_COLLECTIVE(    Collective::ALLREDUCE, Al::NCCLBackend);
//ADD_ALUMINUM_COLLECTIVE(    Collective::BROADCAST, Al::NCCLBackend);
//ADD_ALUMINUM_COLLECTIVE(       Collective::REDUCE, Al::NCCLBackend);
#ifndef HYDROGEN_HAVE_AL_MPI_CUDA
ADD_ALUMINUM_COLLECTIVE(Collective::REDUCESCATTER, Al::NCCLBackend);
#endif // !HYDROGEN_HAVE_AL_MPI_CUDA
#endif // HYDROGEN_HAVE_NCCL2
#endif

#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
// MPICUDA backend only supports AllReduce
//ADD_ALUMINUM_COLLECTIVE(    Collective::ALLREDUCE, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::ALLTOALL, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::GATHER, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(      Collective::SCATTER, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(     Collective::SENDRECV, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(Collective::REDUCESCATTER, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::ALLGATHER, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(    Collective::BROADCAST, Al::MPICUDABackend);
ADD_ALUMINUM_COLLECTIVE(       Collective::REDUCE, Al::MPICUDABackend);
#endif // HYDROGEN_HAVE_AL_MPI_CUDA

template <Device D>
struct BackendsForDeviceT;

template <>
struct BackendsForDeviceT<Device::CPU>
{
    using type = TypeList<Al::MPIBackend>;
};// struct BackendsForDeviceT<Device::CPU>

// Prefer the NCCL2 backend
#ifdef HYDROGEN_HAVE_CUDA
template <>
struct BackendsForDeviceT<Device::GPU>
{
    using type = TypeList<
#ifdef HYDROGEN_HAVE_NCCL2
      Al::NCCLBackend
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
      ,
#endif // HYDROGEN_HAVE_AL_MPI_CUDA
#endif // HYDROGEN_HAVE_NCCL2
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
      Al::MPICUDABackend
#endif // HYDROGEN_HAVE_AL_MPI_CUDA
      >;
};// struct BackendsForDeviceT<Device::GPU>
#endif // HYDROGEN_HAVE_CUDA

// Helper using statement
template <Device D>
using BackendsForDevice = typename BackendsForDeviceT<D>::type;

//
// Aluminum-specific predicates/metafunctions
//

template <typename T, Collective C, typename BackendT>
struct AluminumSupportsBackendAndCollective
    : And<IsAlTypeT<T,BackendT>,IsBackendSupported<C,BackendT>>
{};

template <typename T, Collective C, typename BackendList>
struct IsBackendSupportedByAny
    : Or<AluminumSupportsBackendAndCollective<T,C,Head<BackendList>>,
         IsBackendSupportedByAny<T,C,Tail<BackendList>>>
{};

template <typename T, Collective C>
struct IsBackendSupportedByAny<T,C,TypeList<>>
    : std::false_type
{};

template <typename T, Device D, Collective C>
struct IsAluminumSupported
    : IsBackendSupportedByAny<T,C,BackendsForDevice<D>>
{};


template <typename List, typename U,
          Collective C, template <class,Collective,class> class Pred>
struct SelectFirstOkBackend
    : std::conditional<Pred<U,C,Head<List>>::value,
                       HeadT<List>,
                       SelectFirstOkBackend<Tail<List>,U,C,Pred>>::type
{};

// The "best" backend is the first one in the list that supports our
// type T and implements our collective C.
template <typename T, Device D, Collective C>
struct BestBackendT
    : SelectFirstOkBackend<BackendsForDevice<D>,T,C,
                           AluminumSupportsBackendAndCollective>
{};

template <typename T, Device D, Collective C>
using BestBackend = typename BestBackendT<T,D,C>::type;

#endif // ndefined(HYDROGEN_HAVE_ALUMINUM)

} // namespace El

#endif // HYDROGEN_IMPORTS_ALUMINUM_HPP_
