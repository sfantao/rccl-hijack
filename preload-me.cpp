#if 0

$ROCM_PATH/llvm/bin/clang++ \
    -std=c++20 \
    -I${ROCM_PATH}/include \
    -I${CRAY_MPICH_PREFIX}/include \
    -D__HIP_PLATFORM_AMD__ \
    -fPIC -shared -g -O3 \
    -L${CRAY_MPICH_PREFIX}/lib \
    -Wl,-rpath=${CRAY_MPICH_PREFIX}/lib \
    -lmpi_cray \
    -o libpreload-me.so preload-me.cpp

exit 0

#endif

#include <mpi.h>

#include <stdio.h>
#include <dlfcn.h>
#include <cassert>



// #include <hip/hip_runtime.h>

#define ncclSuccess 0
#define ncclInternalError 3
#define ncclResult_t int
#define ncclDataType_t int
#define ncclComm_t void*
#define ncclRedOp_t int

#define hipStream_t void*

namespace {

    bool is_mpi_initialized_here = false;

// Initializes the symbol of the original runtime symbol and return 0 if success
template<typename T>
int lazy_init(T *&fptr, const char *name) {

    if (!is_mpi_initialized_here) {

        int initialized_before = 0;
        if (MPI_Initialized(&initialized_before))
            return -1;

        if (!initialized_before)
            if (MPI_Init(NULL,NULL))
                return -1;

        is_mpi_initialized_here = 1;
    }

    if(MPI_Barrier(MPI_COMM_WORLD))
        return -1;

    void *&ptr = reinterpret_cast<void *&>(fptr);

    if (ptr) return 0;

    ptr = dlsym(RTLD_NEXT, name);

    assert(ptr);

    return ptr ? 0 : -1;
}

//ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclReduce_orig)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclBcast_orig)(void*, size_t, ncclDataType_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclBroadcast_orig)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclAllReduce_orig)(const void*, void*, size_t,ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclReduceScatter_orig)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclAllGather_orig)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclSend_orig)(const void*, size_t, ncclDataType_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclRecv_orig)(void*, size_t, ncclDataType_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclGather_orig)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclScatter_orig)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclAllToAll_orig)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclAllToAllv(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[], void *recvbuff, const size_t recvcounts[], const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
ncclResult_t (*ncclAllToAllv_orig)(const void *, const size_t [], const size_t [], void *, const size_t [], const size_t [], ncclDataType_t, ncclComm_t, hipStream_t) = nullptr;
//ncclResult_t  ncclGroupStart();
ncclResult_t (*ncclGroupStart_orig)() = nullptr;
//ncclResult_t  ncclGroupEnd();
ncclResult_t (*ncclGroupEnd_orig)() = nullptr;

size_t hipMallocCount = 0;

} // namespace

extern "C" {
ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclReduce_orig, "ncclReduce")) return ncclInternalError;

    return ncclReduce_orig(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}
ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclBcast_orig, "ncclBcast")) return ncclInternalError;

    return ncclBcast_orig(buff, count, datatype, root, comm, stream);
}
ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclBroadcast_orig, "ncclBroadcast")) return ncclInternalError;

    return ncclBroadcast_orig(sendbuff, recvbuff, count, datatype, root, comm, stream);
}
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclAllReduce_orig, "ncclAllReduce")) return ncclInternalError;

    printf("sfantao --> Hello from ncclAllReduce!!!\n");

    return ncclAllReduce_orig(sendbuff, recvbuff, count, datatype, op, comm, stream);
}
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclReduceScatter_orig, "ncclReduceScatter")) return ncclInternalError;

    return ncclReduceScatter_orig(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclAllGather_orig, "ncclAllGather")) return ncclInternalError;

    return ncclAllGather_orig(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclSend_orig, "ncclSend")) return ncclInternalError;

    return ncclSend_orig(sendbuff, count, datatype, peer, comm, stream);
}
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclRecv_orig, "ncclRecv")) return ncclInternalError;

    return ncclRecv_orig(recvbuff, count, datatype, peer, comm, stream);
}
ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclGather_orig, "ncclGather")) return ncclInternalError;

    return ncclGather_orig(sendbuff, recvbuff, sendcount, datatype, root, comm, stream);
}
ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclScatter_orig, "ncclScatter")) return ncclInternalError;

    return ncclScatter_orig(sendbuff, recvbuff, recvcount, datatype, root, comm, stream);
}
ncclResult_t  ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclAllToAll_orig, "ncclAllToAll")) return ncclInternalError;

    return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}
ncclResult_t  ncclAllToAllv(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[], void *recvbuff, const size_t recvcounts[], const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
    
    if(lazy_init(ncclAllToAllv_orig, "ncclAllToAllv")) return ncclInternalError;

    return ncclAllToAllv_orig(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype, comm, stream);
}
ncclResult_t  ncclGroupStart(){
    
    if(lazy_init(ncclGroupStart_orig, "ncclGroupStart")) return ncclInternalError;

    return ncclGroupStart_orig();
}
ncclResult_t  ncclGroupEnd(){
    
    if(lazy_init(ncclGroupEnd_orig, "ncclGroupEnd")) return ncclInternalError;

    return ncclGroupEnd_orig();
}
} // extern "C" 

