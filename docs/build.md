# Dependencies

Dependency | Version | Required           | Runtime
---        | ---     | ---                | ---
Kokkos     | 3.1.00  | :heavy_check_mark: | :heavy_check_mark:
MPI        | 2       |                    | :heavy_check_mark:
CMake      | 3.12    | :heavy_check_mark: |

# Installation

## Installation using Spack

ArborX can be installed using [Spack](https://github.com/spack/spack) package manager:
```shell
spack install arborx
```
The package provides several options to control the configuration. For example,
to install ArborX with CUDA backend but without MPI, do
```shell
spack install arborx~mpi+cuda
```

## Manual installation

The dependencies may be installed using [Spack](https://github.com/spack/spack)
package manager, or manually. Of note:
- If Kokkos is being installed as part of [Trilinos](https://github.com/trilinos/Trilinos),
  set `-DCMAKE_CXX_STANDARD=14`. Please note that this will only compile the
  Kokkos package with `C++14` standard. If you want to compile the full
  Trilinos with C++14, you would need a second option
  `-DTrilinos_CXX11_FLAGS="-std=c++14"`
- The list of backends supported by ArborX will be set to the ones provided by
  the Kokkos

Once the dependencies are installed, configure ArborX:
- Add `-DCMAKE_PREFIX_PATH="$KOKKOS_DIR"`
- If Kokkos' CUDA backend is enabled, add `-DCMAKE_CXX_COMPILER="nvcc_wrapper"`
- If MPI is desired, add `-DARBORX_ENABLE_MPI=ON`

# Building against ArborX

For projects using CMake, add to `CMakeLists.txt`
```CMake
find_package(ArborX REQUIRED)
```
For any targets requiring ArborX, add
```CMake
target_link_libraries(<target> ArborX::ArborX)
```
