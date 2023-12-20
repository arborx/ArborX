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

#include <ArborX_BruteForce.hpp>
#include <ArborX_HyperPoint.hpp>
#include <ArborX_HyperSphere.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include "brute_force_vs_bvh.hpp"

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

namespace ArborXBenchmark
{
template <int DIM, typename FloatingPoint>
struct Placeholder
{
  int count;
};
} // namespace ArborXBenchmark

// Primitives are a set of points located at (i, i, i),
// with i = 0, ..., n-1
template <int DIM, typename FloatingPoint>
struct ArborX::AccessTraits<ArborXBenchmark::Placeholder<DIM, FloatingPoint>,
                            ArborX::PrimitivesTag>
{
  using Primitives = ArborXBenchmark::Placeholder<DIM, FloatingPoint>;
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Primitives d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Primitives, size_type i)
  {
    ArborX::ExperimentalHyperGeometry::Point<DIM, FloatingPoint> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = i;
    return point;
  }
};

// Predicates are sphere intersections with spheres of radius i
// centered at (i, i, i), with i = 0, ..., n-1
template <int DIM, typename FloatingPoint>
struct ArborX::AccessTraits<ArborXBenchmark::Placeholder<DIM, FloatingPoint>,
                            ArborX::PredicatesTag>
{
  using Predicates = ArborXBenchmark::Placeholder<DIM, FloatingPoint>;
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Predicates d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Predicates, size_type i)
  {
    ArborX::ExperimentalHyperGeometry::Point<DIM, FloatingPoint> center;
    for (int d = 0; d < DIM; ++d)
      center[d] = i;
    return attach(
        intersects(
            ArborX::ExperimentalHyperGeometry::Sphere<DIM, FloatingPoint>{
                center, (FloatingPoint)i}),
        i);
  }
};

namespace ArborXBenchmark
{

template <int DIM, typename FloatingPoint>
static void run_fp(int nprimitives, int nqueries, int nrepeats)
{
  ExecutionSpace space{};
  Placeholder<DIM, FloatingPoint> primitives{nprimitives};
  Placeholder<DIM, FloatingPoint> predicates{nqueries};

  using Point = ArborX::ExperimentalHyperGeometry::Point<DIM, FloatingPoint>;

  for (int i = 0; i < nrepeats; i++)
  {
    [[maybe_unused]] unsigned int out_count;
    {
      Kokkos::Timer timer;
      ArborX::BoundingVolumeHierarchy<MemorySpace,
                                      ArborX::Details::PairIndexVolume<Point>>
          bvh{space, ArborX::Details::LegacyValues<decltype(primitives), Point>{
                         primitives}};

      Kokkos::View<int *, ExecutionSpace> indices("Benchmark::indices_ref", 0);
      Kokkos::View<int *, ExecutionSpace> offset("Benchmark::offset_ref", 0);
      bvh.query(space, predicates, ArborX::Details::LegacyDefaultCallback{},
                indices, offset);

      space.fence();
      double time = timer.seconds();
      if (i == 0)
        printf("Collisions: %.5f\n",
               (float)(indices.extent(0)) / (nprimitives * nqueries));
      printf("Time BVH  : %lf\n", time);
      out_count = indices.extent(0);
    }

    {
      Kokkos::Timer timer;
      ArborX::BruteForce<MemorySpace, ArborX::Details::PairIndexVolume<Point>>
          brute{space,
                ArborX::Details::LegacyValues<decltype(primitives), Point>{
                    primitives}};

      Kokkos::View<int *, ExecutionSpace> indices("Benchmark::indices", 0);
      Kokkos::View<int *, ExecutionSpace> offset("Benchmark::offset", 0);
      brute.query(space, predicates, ArborX::Details::LegacyDefaultCallback{},
                  indices, offset);

      space.fence();
      double time = timer.seconds();
      printf("Time BF   : %lf\n", time);
      assert(out_count == indices.extent(0));
    }
  }
}

template <int DIM>
void run(int nprimitives, int nqueries, int nrepeats)
{
  printf("Dimension : %d\n", DIM);
  printf("Primitives: %d\n", nprimitives);
  printf("Predicates: %d\n", nqueries);
  printf("Iterations: %d\n", nrepeats);

  printf("-------------------\n");
  printf("Precision : float\n");
  run_fp<DIM, float>(nprimitives, nqueries, nrepeats);
  printf("-------------------\n");
  printf("Precision : double\n");
  run_fp<DIM, double>(nprimitives, nqueries, nrepeats);
}

} // namespace ArborXBenchmark
