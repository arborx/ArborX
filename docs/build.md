# Installation

## Installation using Spack

The simplest way to install ArborX is to use [Spack](https://github.com/spack/spack) package manager:
```shell
spack install arborx
```

## Manual installation

ArborX has the following dependencies:

Dependency | Version | Required
--- | --- | ---
Kokkos | 2.8.00 | :heavy_check_mark:
MPI    | 2      |
CMake  | 3.14   | :heavy_check_mark:

The dependencies may be installed using [Spack](https://github.com/spack/spack) package manager:
FIXME (try out Kokkos install)
```shell
spack install kokkos
spack install mpi
```
or manually (cmake and mpi - easy, Kokkos is here):
FIXME: do we need instructions to install Kokkos here?

Once the dependencies are installed, configure ArborX with
```shell
#!/usr/bin/env bash
ARGS=(
    -D CMAKE_BUILD_TYPE=RELWITHDEBINFO
    -D CMAKE_INSTALL_PREFIX="$ARBORX_INSTALL_DIR"
    -D BUILD_SHARED_LIBS=ON

    ### TPLs ###
    -D ARBORX_ENABLE_MPI=ON
    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR"

    ### COMPILERS AND FLAGS ###
    -D CMAKE_CXX_COMPILER="nvcc_wrapper"
    )
cmake "${ARGS[@]}" "${ARBORX_DIR}"
```
If building with CUDA enabled ArborX/Kokkos, add
```
-D CMAKE_CXX_COMPILER="nvcc_wrapper"
```
to `ARGS`.

# Building against ArborX

For projects using CMake, add to `CMakeLists.txt`
```CMake
find_package(ArborX REQUIRED)
```
For any targets requiring ArborX, add
```CMake
target_link_libraries(<target> ArborX::ArborX)
```
