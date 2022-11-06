find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# options
# gui option
option(GUI "OPENGL Rendering" OFF)

# omp flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# include & requirements
set(THREADS_PREFER_PTHREAD_FLAG ON)
include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
)
link_libraries(
    ${MPI_LIBRARIES}
    ${CUDA_LIBRAIRES}
)


# opengl & glut
if(GUI)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_DINCLUDE_DIRS})
    add_definitions(-DGUI)
    link_libraries(
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
    )
endif()

# targets & libs
add_executable(main main.cpp)


