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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <random>
#include <vector>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct FirstOctant
{};

struct NearestToOrigin
{
  int k;
};

template <>
struct ArborX::AccessTraits<FirstOctant>
{
  static KOKKOS_FUNCTION std::size_t size(FirstOctant) { return 1; }
  static KOKKOS_FUNCTION auto get(FirstOctant, std::size_t)
  {
    return ArborX::intersects(ArborX::Box<3>{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}});
  }
  using memory_space = MemorySpace;
};

template <>
struct ArborX::AccessTraits<NearestToOrigin>
{
  static KOKKOS_FUNCTION std::size_t size(NearestToOrigin) { return 1; }
  static KOKKOS_FUNCTION auto get(NearestToOrigin d, std::size_t)
  {
    return ArborX::nearest(ArborX::Point{0.f, 0.f, 0.f}, d.k);
  }
  using memory_space = MemorySpace;
};

struct PrintfCallback
{
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, Value const &value,
                                  OutputFunctor const &out) const
  {
    auto const index = value.index;
    Kokkos::printf("Found %d from functor\n", index);
    out(index);
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using Point = ArborX::Point<3>;

  int const n = 100;
  std::vector<Point> points;
  // Fill vector with random points in [-1, 1]^3
  std::uniform_real_distribution<float> dis{-1., 1.};
  std::default_random_engine gen;
  auto rd = [&]() { return dis(gen); };
  std::generate_n(std::back_inserter(points), n, [&]() {
    return Point{rd(), rd(), rd()};
  });

  ArborX::BoundingVolumeHierarchy bvh{
      ExecutionSpace{},
      ArborX::Experimental::attach_indices(Kokkos::create_mirror_view_and_copy(
          MemorySpace{},
          Kokkos::View<Point *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
              points.data(), points.size())))};

  {
    Kokkos::View<int *, MemorySpace> values("Example::values", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
    ArborX::query(bvh, ExecutionSpace{}, FirstOctant{}, PrintfCallback{},
                  values, offsets);
#ifndef __NVCC__
    ArborX::query(
        bvh, ExecutionSpace{}, FirstOctant{},
        KOKKOS_LAMBDA(auto /*predicate*/, auto value, auto /*output_functor*/) {
          Kokkos::printf("Found %d from generic lambda\n", value.index);
        },
        values, offsets);
#endif
  }

  {
    int const k = 10;
    Kokkos::View<int *, MemorySpace> values("Example::values", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
    ArborX::query(bvh, ExecutionSpace{}, NearestToOrigin{k}, PrintfCallback{},
                  values, offsets);
#ifndef __NVCC__
    ArborX::query(
        bvh, ExecutionSpace{}, NearestToOrigin{k},
        KOKKOS_LAMBDA(auto /*predicate*/, auto value, auto /*output_functor*/) {
          Kokkos::printf("Found %d from generic lambda\n", value.index);
        },
        values, offsets);
#endif
  }

  {
    // EXPERIMENTAL
    Kokkos::View<int, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> c(
        "counter");

#ifndef __NVCC__
    bvh.query(
        ExecutionSpace{}, FirstOctant{},
        KOKKOS_LAMBDA(auto /*predicate*/, auto value) {
          Kokkos::printf("%d %d %d\n", ++c(), -1, value.index);
        });
#endif
  }

  return 0;
}
