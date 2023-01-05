/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>

#include <boost/program_options.hpp>

#include <benchmark/benchmark.h>

struct UnweightedEdge
{
  unsigned int source;
  unsigned int target;
};

struct AllowLoops
{};
struct DisallowLoops
{};

template <typename ExecutionSpace>
Kokkos::View<UnweightedEdge *, typename ExecutionSpace::memory_space>
buildEdges(AllowLoops, ExecutionSpace const &exec_space, int num_edges)
{
  using MemorySpace = typename ExecutionSpace::memory_space;
  Kokkos::View<UnweightedEdge *, MemorySpace> edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Benchmark::edges"),
      num_edges);

  Kokkos::Random_XorShift1024_Pool<ExecutionSpace> rand_pool(1984);
  Kokkos::parallel_for(
      "ArborX::Benchmark::init_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(unsigned i) {
        auto rand_gen = rand_pool.get_state();
        do
        {
          edges(i) = {rand_gen.urand() % num_edges,
                      rand_gen.urand() % num_edges};
        } while (edges(i).source == edges(i).target); // no self loops
        rand_pool.free_state(rand_gen);
      });
  return edges;
}

template <typename ExecutionSpace>
Kokkos::View<UnweightedEdge *, typename ExecutionSpace::memory_space>
buildEdges(DisallowLoops, ExecutionSpace const &exec_space, int num_edges)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::Random_XorShift1024_Pool<ExecutionSpace> rand_pool(1984);
  // Construct random permutation by sorting a vector of random values
  Kokkos::View<int *, MemorySpace> random_values(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Benchmark::random_values"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Benchmark::init_random_values",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        random_values(i) = rand_gen.rand();
        rand_pool.free_state(rand_gen);
      });
  auto permute = ArborX::Details::sortObjects(exec_space, random_values);

  // Init edges in a random order
  Kokkos::View<UnweightedEdge *, MemorySpace> edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Benchmark::edges"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Benchmark::init_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(unsigned i) {
        auto rand_gen = rand_pool.get_state();
        edges(permute(i)) = {rand_gen.urand() % (i + 1), i + 1};
        rand_pool.free_state(rand_gen);
      });
  return edges;
}

template <typename ExecutionSpace>
auto buildUnionFind(ExecutionSpace const &exec_space, int n)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Benchmark::labels"),
      n);
  ArborX::iota(exec_space, labels);
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Serial>)
    return ArborX::Details::UnionFind<MemorySpace, /*DoSerial*/ true>(labels);
  else
#endif
    return ArborX::Details::UnionFind<MemorySpace, /*DoSerial*/ false>(labels);
}

template <typename Tag>
void BM_union_find(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  ExecutionSpace exec_space;

  auto const num_edges = state.range(0);
  auto const n = num_edges + 1;

  auto edges = buildEdges(Tag{}, exec_space, num_edges);
  auto union_find = buildUnionFind(exec_space, n);

  for (auto _ : state)
  {
    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_for(
        "ArborX::Benchmark::union-find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
        KOKKOS_LAMBDA(int e) {
          int i = edges(e).source;
          int j = edges(e).target;

          union_find.merge(i, j);
        });

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] = benchmark::Counter(
      num_edges, benchmark::Counter::kIsIterationInvariantRate);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

  benchmark::Initialize(&argc, argv);

  BENCHMARK_TEMPLATE1(BM_union_find, AllowLoops)
      ->RangeMultiplier(10)
      ->Range(10000, 100000)
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  BENCHMARK_TEMPLATE1(BM_union_find, DisallowLoops)
      ->RangeMultiplier(10)
      ->Range(10000, 100000)
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
