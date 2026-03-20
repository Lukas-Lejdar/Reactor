#set(CMAKE_CXX_STANDARD 20)
#set(TARGET "step20")
#set(TARGET_SRC src/mixed_fem.cc)
#project(${TARGET})

set(TARGET "mixed_fem")
set(TARGET_SRC src/main.cc)

cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.7.1 HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
if(NOT ${deal_II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II")
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reactor_solutions_mixed)

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()
