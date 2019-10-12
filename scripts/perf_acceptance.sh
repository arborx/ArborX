#!/usr/bin/env bash

# Fail if any command fails
set -e

HELP_STRING="Usage: $0 <branch> [<extra_cmake_args>]"
if [[ $# -lt 1 ]]; then
  echo -e "$HELP_STRING"
  exit 0
fi
BRANCHES=("master" "$1")
shift

BENCHMARK_CONFIGS=(
  "--benchmark_filter=\"Cuda\" --benchmark_repetitions=10 --benchmark_display_aggregates_only=false"
  "--benchmark_filter=\"Serial\" --benchmark_repetitions=10 --benchmark_display_aggregates_only=false"
)

# Default settings
CMAKE_DEFAULT_CONFIG=(
    -D CMAKE_BUILD_TYPE=Release
    -D BUILD_SHARED_LIBS=ON
    -D ARBORX_ENABLE_MPI=ON
    -D ARBORX_ENABLE_BENCHMARKS=ON
    -D ARBORX_ENABLE_EXAMPLES=OFF
    -D ARBORX_ENABLE_TESTS=OFF
    -D CMAKE_CXX_COMPILER="nvcc_wrapper"
)
# Set CMAKE_EXTRA_CONFIG array in an external file for specific configuration of
# dependencies
if [ -e ".perf_tpl_config" ]; then
  source .perf_tpl_config
fi

ARBORX_SOURCE_DIR=$(git rev-parse --show-toplevel)

PERF_DIR="$(pwd)/perf_results"

mkdir "$PERF_DIR"
cd "$PERF_DIR"

# Make sure the tree is not dirty
git_description="$(git describe --long --dirty --tags)"
if [[ "$git_description" == *"dirty" ]]; then
  echo "Dirty git directory, aborting..."
  exit 1
fi

for build_number in $(seq 0 $((${#BRANCHES[@]}-1))); do
  echo "build #${build_number}"

  git checkout "${BRANCHES[$build_number]}"
  echo "git hash: $(git log --pretty=format:%h -n 1)"

  BUILD_DIR="$PERF_DIR/build${build_number}"
  mkdir "$BUILD_DIR" && cd "$BUILD_DIR"

  cmake_cmd="cmake ${CMAKE_DEFAULT_CONFIG[@]} ${CMAKE_EXTRA_CONFIG[@]} ${ARBORX_SOURCE_DIR}"
  output="configure.log"
  echo "$cmake_cmd" | tee    "$output"
  eval "$cmake_cmd" 2>&1 | tee -a "$output"
  make -j | tee make.log

  cd benchmarks/bvh_driver
  for i in $(seq 0 $((${#BENCHMARK_CONFIGS[@]}-1))); do
    test_cmd="./ArborX_BoundingVolumeHierarchy.exe ${BENCHMARK_CONFIGS[$i]}"
    output="$PERF_DIR/build${build_number}_config${i}.log"
    echo "$test_cmd" | tee    "$output"
    eval "$test_cmd" 2>&1 | tee -a "$output"
  done
  cd "$PERF_DIR"
done

# Compare results
