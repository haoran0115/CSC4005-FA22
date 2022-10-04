find_package(MPI REQUIRED)

# cmake includes
include_directories(${MPI_INCLUDE_PATH})

# executable
add_executable(main main.cpp)
add_executable(main.seq main.seq.cpp)
target_link_libraries(main ${MPI_LIBRARIES})

