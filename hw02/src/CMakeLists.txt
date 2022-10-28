find_package(MPI REQUIRED)

# options
# gui option
option(GUI "OPENGL Rendering" OFF)

# include & requirements
# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
# mpi
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
# opengl & glut
if(GUI)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_DINCLUDE_DIRS})
    add_definitions(-DGUI)
endif()

# executable
add_executable(main.seq main.seq.cpp)
add_executable(main.pthread main.pthread.cpp)
add_executable(main.pthread_ds main.pthread_ds.cpp)
add_executable(main.mpi main.mpi.cpp)
target_link_libraries(main.pthread Threads::Threads)
target_link_libraries(main.pthread_ds Threads::Threads)
target_link_libraries(main.mpi ${MPI_LIBRARIES})
if(GUI)
    target_link_libraries(main.seq ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
    target_link_libraries(main.mpi ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
    target_link_libraries(main.pthread ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
    target_link_libraries(main.pthread_ds ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
endif()

