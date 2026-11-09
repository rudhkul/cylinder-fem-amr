cmake_minimum_required(VERSION 3.24)
project(cylinder_fem_amr LANGUAGES CXX CUDA)

# GPU target: native by default; override with -DCMAKE_CUDA_ARCHITECTURES=86 (etc.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

find_package(OpenMP REQUIRED COMPONENTS CXX)

function(cfa_target name source)
  add_executable(${name} ${source})
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,${OpenMP_CXX_FLAGS}>)
  target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

cfa_target(cylinder_fem_amr src/cylinder_fem_amr.cu)
cfa_target(test_cfa tests/test_cfa.cu)

enable_testing()
add_test(NAME unit COMMAND test_cfa)
add_test(NAME smoke COMMAND cylinder_fem_amr --steps 30 --prefix ctest_smoke)
