#!/bin/bash

EXTRA_ARGS=("$@")

rm -f  CMakeCache.txt
rm -rf CMakeFiles/

ARGS=(
    -D CMAKE_BUILD_TYPE=Debug
    -D BUILD_SHARED_LIBS=ON

    ### TPLs
    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BENCHMARK_DIR;$BOOST_DIR"

    ### COMPILERS AND FLAGS ###
    -D CMAKE_CXX_COMPILER="$KOKKOS_DIR/bin/nvcc_wrapper"
    -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"

    ### MISC ###
    -D MPIEXEC_PREFLAGS="--allow-run-as-root"
)

cd $ARBORX_DIR
rm -rf build
mkdir build
cd build

cmake "${ARGS[@]}" "${EXTRA_ARGS[@]}" "${ARBORX_DIR}"

make -j${NPROCS}

ctest --no-compress-output -T Test
