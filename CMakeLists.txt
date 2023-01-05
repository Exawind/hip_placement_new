cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project(HIP_Placement_New CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL
  "Create compile commands database")

option(USE_EXTERNAL_KOKKOS "Use external Kokkos installation" OFF)
option(TEST_PLACEMENT_NEW "Activate testing placement new functions" ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos
  GIT_TAG develop
  )

if (USE_EXTERNAL_KOKKOS)
  find_package(Kokkos REQUIRED)
else()
  FetchContent_MakeAvailable(kokkos)
endif()

add_executable(hip_placement_new)
add_subdirectory(src)
