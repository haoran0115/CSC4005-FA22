find_package(MPI REQUIRED)

# cmake includes
include_directories(${MPI_INCLUDE_PATH})
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# executable
add_executable(main.seq main.seq.cpp)
add_executable(main.pthread main.pthread.cpp)
add_executable(main.mpi main.mpi.cpp)
target_link_libraries(main.pthread PRIVATE Threads::Threads)
target_link_libraries(main.mpi ${MPI_LIBRARIES})

