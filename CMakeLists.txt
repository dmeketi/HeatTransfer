cmake_minimum_required(VERSION 3.20)
project(HeatTransfer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(Kokkos REQUIRED)

add_executable(heat

        rowmaj.cpp
)
target_link_libraries(heat PRIVATE Kokkos::kokkos)