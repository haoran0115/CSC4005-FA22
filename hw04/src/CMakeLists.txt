find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# options
# gui option
option(GUI "OPENGL Rendering" OFF)

# omp flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# libraries
add_library(cudalib cudalib.cu)
set(THREADS_PREFER_PTHREAD_FLAG ON)
include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
)
link_libraries(
    ${MPI_LIBRARIES}
    ${CUDA_LIBRAIRES}
    cudalib
)


# targets & libs
add_executable(main.seq main.cpp)
add_executable(main.omp main.cpp)
add_executable(main.pth main.cpp)
# add_executable(main.cu main.cpp)
# add_executable(main.mpi main.mpi.cpp)
target_compile_definitions(main.omp PUBLIC OMP)
target_compile_definitions(main.pth PUBLIC PTH)
# target_compile_definitions(main.cu PUBLIC CUDA)

# opengl & glut
if(GUI)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_DINCLUDE_DIRS})
    link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
    add_executable(main.seq.gui main.cpp)
    add_executable(main.omp.gui main.cpp)
    add_executable(main.pth.gui main.cpp)
    # add_executable(main.cu.gui main.cpp)
    # add_executable(main.mpi.gui main.mpi.cpp)
    target_compile_definitions(main.seq.gui PUBLIC GUI)
    target_compile_definitions(main.omp.gui PUBLIC GUI OMP)
    target_compile_definitions(main.pth.gui PUBLIC GUI PTH)
    # target_compile_definitions(main.mpi.gui PUBLIC GUI)
    # target_compile_definitions(main.cu.gui PUBLIC GUI CUDA)
endif()


