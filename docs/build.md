# Dependencies

Dependency | Version | Required           | Runtime
---        | ---     | ---                | ---
[Kokkos](https://github.com/kokkos/kokkos)     | 3.1.00  | :heavy_check_mark: | :heavy_check_mark:
MPI        | 2       |                    | :heavy_check_mark:
CMake      | 3.12    | :heavy_check_mark: |

# Build and installation

The build instructions for the Kokkos library can be found
[here](https://github.com/kokkos/kokkos#building-and-installing-kokkos).
ArborX requires Kokkos CMake build to have `Kokkos_ENABLE_CUDA_LAMBDA=ON` if
`Kokkos_ENABLE_CUDA=ON`.

For example, to build Kokkos with three backends (Serial, OpenMP and CUDA) for
POWER9 CPU and Nvidia Volta GPU (CUDA CC 7.0), configure it with
```CMake
OPTIONS=(
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_DIR}"
    -D CMAKE_CXX_COMPILER="${KOKKOS_SOURCE_DIR}/bin/nvcc_wrapper"
    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ENABLE_OPENMP=ON
    -D Kokkos_ENABLE_CUDA=ON
        -D Kokkos_ENABLE_CUDA_LAMBDA=ON
    -D Kokkos_ARCH_POWER9=ON
    -D Kokkos_ARCH_Volta70=ON
    )
cmake "${OPTIONS[@]}" "${KOKKOS_SOURCE_DIR:-../}"
```

## CMake

Assuming that Kokkos library is installed in `$KOKKOS_INSTALL_DIR`, configure
ArborX as
```CMake
OPTIONS=(
    -D CMAKE_INSTALL_PREFIX="${ARBORX_INSTALL_DIR}"
    -D ARBORX_ENABLE_MPI=ON
    -D CMAKE_PREFIX_PATH="${KOKKOS_INSTALL_DIR}"
    -D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper"
    -D CMAKE_CXX_EXTENSIONS=OFF # required by Kokkos
    )
cmake "${OPTIONS[@]}" "${ARBORX_DIR:-../}"
```
then run `make install`. ArborX will then be installed in
`$ARBORX_INSTALL_DIR`. If Kokkos was built without CUDA support, remove the
`CMAKE_CXX_COMPILER` configuration option.

To disable MPI, configure with
```CMake
    -D ARBORX_ENABLE_MPI=OFF
```

To enable ArborX examples, configure with
```CMake
    -D ARBORX_ENABLE_EXAMPLES=ON
```

To validate the ArborX build, install Boost version 1.67 or 1.68
(`$BOOST_INSTALL_DIR`), configure with
```CMake
    -D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR;$BOOST_INSTALL_DIR"
    -D ARBORX_ENABLE_TESTS=ON
```
and run `ctest` after completing the build.

To enable ArborX benchmarks, install [Google
Benchmark](https://github.com/google/benchmark) version 1.4 or later
(`$BENCHMARK_INSTALL_DIR`), Boost version 1.67 or later (`$BOOST_INSTALL_DIR`),
configure with
```CMake
    -D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR;$BOOST_INSTALL_DIR;$BENCHMARK_INSTALL_DIR"
    -D ARBORX_ENABLE_BENCHMARKS=ON
```
The individual benchmarks can then be run from `benchmarks` directory.

ArborX also supports building against Kokkos built as part of
[Trilinos](https://github.com/trilinos/Trilinos). This requires Trilinos hash
[de15ca5352](https://github.com/trilinos/Trilinos/commit/de15ca5352bdd31e94a2ba6af1a1b56cefe546da)
or later. In this case, `$KOKKOS_INSTALL_DIR` should point to the Trilinos installation
directory.

## Spack

ArborX can also be installed using the [Spack](https://github.com/spack/spack)
package manager. A basic installation can be done as:
```shell
spack install arborx
```
Spack allows options and compilers to be turned on in the install command:
```shell
spack install arborx@1.0 %gcc@8.1.0 ~mpi+cuda
```
This example illustrates the most common parameters:
- Variants are specified with `~` or `+` (e.g. `+cuda`) and enable or disable certain options
- Version (`@version`) immediately follows `arborx` and can specify a specific ArborX version
- Compiler (`%compiler`) can be set if an exact compiler is desired; otherwise, a default compiler is chosen

For a complete list of available ArborX options, run:
```shell
spack info arborx
```

# Building against ArborX

For projects using CMake, add to `CMakeLists.txt`
```CMake
find_package(ArborX REQUIRED)
```
For any targets requiring ArborX, add
```CMake
target_link_libraries(<target> ArborX::ArborX)
```
Note that because of the way CMake deals with dependencies, users of ArborX
will need to make sure that Kokkos installation directory is visible to CMake.
The easiest way to address it is to add both to the configuration options:
```CMake
    -D CMAKE_PREFIX_PATH="$ARBORX_INSTALL_DIR;$KOKKOS_INSTALL_DIR"
```
