/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_BruteForce.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Core.hpp>

#include "brute_force_vs_bvh.hpp"

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

namespace ArborXBenchmark
{
struct PrimitivesTag
{};
struct PredicatesTag
{};

template <int DIM, typename FloatingPoint, typename Tag>
struct Placeholder
{
  int count;
};
} // namespace ArborXBenchmark

template <int DIM, typename FloatingPoint, typename Tag>
struct ArborX::AccessTraits<
    ArborXBenchmark::Placeholder<DIM, FloatingPoint, Tag>>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;

  static KOKKOS_FUNCTION size_type
  size(ArborXBenchmark::Placeholder<DIM, FloatingPoint, Tag> d)
  {
    return d.count;
  }

  static KOKKOS_FUNCTION auto
  get(ArborXBenchmark::Placeholder<DIM, FloatingPoint,
                                   ArborXBenchmark::PrimitivesTag>,
      size_type i)
  {
    // Primitives are a set of points located at (i, i, i),
    // with i = 0, ..., n-1
    ArborX::Point<DIM, FloatingPoint> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = i;
    return point;
  }

  static KOKKOS_FUNCTION auto
  get(ArborXBenchmark::Placeholder<DIM, FloatingPoint,
                                   ArborXBenchmark::PredicatesTag>,
      size_type i)
  {
    // Predicates are sphere intersections with spheres of radius i
    // centered at (i, i, i), with i = 0, ..., n-1
    ArborX::Point<DIM, FloatingPoint> center;
    for (int d = 0; d < DIM; ++d)
      center[d] = i;
    return attach(intersects(ArborX::Sphere{center, (FloatingPoint)i}), i);
  }
};

namespace ArborXBenchmark
{

template <int DIM, typename FloatingPoint>
static void run_fp(int nprimitives, int nqueries, int nrepeats)
{
  ExecutionSpace space{};

  Placeholder<DIM, FloatingPoint, PrimitivesTag> primitives{nprimitives};
  Placeholder<DIM, FloatingPoint, PredicatesTag> predicates{nqueries};
  using Point = ArborX::Point<DIM, FloatingPoint>;

  for (int i = 0; i < nrepeats; i++)
  {
    [[maybe_unused]] unsigned int out_count;
    {
      Kokkos::Timer timer;
      ArborX::BoundingVolumeHierarchy bvh{space, primitives};

      Kokkos::View<Point *, ExecutionSpace> values("Benchmark::values_ref", 0);
      Kokkos::View<int *, ExecutionSpace> offset("Benchmark::offset_ref", 0);
      bvh.query(space, predicates, values, offset);

      space.fence();
      double time = timer.seconds();
      if (i == 0)
        printf("Collisions: %.5f\n",
               (float)(values.extent(0)) / (nprimitives * nqueries));
      printf("Time BVH  : %lf\n", time);
      out_count = values.extent(0);
    }

    {
      Kokkos::Timer timer;
      ArborX::BruteForce brute{space, primitives};

      Kokkos::View<Point *, ExecutionSpace> values("Benchmark::values", 0);
      Kokkos::View<int *, ExecutionSpace> offset("Benchmark::offset", 0);
      brute.query(space, predicates, values, offset);

      space.fence();
      double time = timer.seconds();
      printf("Time BF   : %lf\n", time);
      assert(out_count == values.extent(0));
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
