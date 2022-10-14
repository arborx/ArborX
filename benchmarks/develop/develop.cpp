/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <benchmark/benchmark.h>

void BM_benchmark(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  Kokkos::View<int *> view(Kokkos::view_alloc(exec_space, "Benchmark::view",
                                              Kokkos::WithoutInitializing),
                           n);

  exec_space.fence();
  for (auto _ : state)
  {
    // This code gets timed
    Kokkos::parallel_for(
        "Benchmark::iota",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) { view(i) = i; });
    exec_space.fence();
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);

  BENCHMARK(BM_benchmark)->RangeMultiplier(10)->Range(100, 10000);

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
