cmake_minimum_required(VERSION 3.6)
project(IonTrap++)

# Notes:
# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

set(CMAKE_CXX_STANDARD 11) # enable C++11 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread -L /usr/local/lib")

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

# Bulding the external libraries
include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt)

include_directories(./experiment/)
file(GLOB EXP_SOURCE_FILES "./experiment/*.h" "./experiment/*.cpp" "./experiment/fidelity/*.h" "./experiment/fidelity/*.cpp")

# ion-trap-qsd-1.3.5
add_executable(IonTrap-QSD ion-trap-qsd-1.3.5/main.cpp ${EXP_SOURCE_FILES})
target_link_libraries(IonTrap-QSD qsd cnpy boost_filesystem boost_serialization Eigen3::Eigen)

# ion-trap-qsd-1.3.5-mpi
#add_executable(IonTrap-QSD-MPI ion-trap-qsd-1.3.5-mpi/main.cpp ${EXP_SOURCE_FILES})
#target_link_libraries(IonTrap-QSD-MPI qsd cnpy boost_serialization boost_mpi)

# ion-trap-odeint
#add_executable(IonTrap-ODE ion-trap-odeint/main.cpp ${EXP_SOURCE_FILES})
#target_link_libraries(IonTrap-ODE qsd cnpy boost_serialization)

# ion-trap-adrianmichel
add_executable(IonTrap-DE ion-trap-adrianmichel/main.cpp ${EXP_SOURCE_FILES})
target_link_libraries(IonTrap-DE qsd de cnpy boost_filesystem boost_serialization boost_thread boost_system Eigen3::Eigen)

# ion-trap-pagmo
#set(Pagmo_DIR path_to_pagmo)
#find_package(Pagmo REQUIRED)

#add_executable(IonTrap-PaGMO ion-trap-pagmo/main.cpp ${EXP_SOURCE_FILES})
#target_link_libraries(IonTrap-PaGMO qsd cnpy boost_serialization boost_thread boost_system Pagmo::pagmo)

# ion-trap-rl
#find_package (Python3 COMPONENTS Interpreter Development NumPy)

#find_package(pybind11)
#include_directories(pybind11)

#add_executable(IonTrap-RL ion-trap-rl/main.cpp ${EXP_SOURCE_FILES})
#target_link_libraries(IonTrap-RL qsd cnpy boost_filesystem boost_serialization Eigen3::Eigen pybind11::embed)