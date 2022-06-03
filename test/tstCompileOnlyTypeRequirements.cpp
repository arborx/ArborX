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

#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

namespace Test
{
// NOTE only supporting Point and Box
using PrimitivePointOrBox = ArborX::Point;
// using PrimitivePointOrBox = ArborX::Box;

// clang-format off
struct FakeBoundingVolume
{
  KOKKOS_FUNCTION FakeBoundingVolume &operator+=(PrimitivePointOrBox) { return *this; }
  KOKKOS_FUNCTION void operator+=(PrimitivePointOrBox) volatile {}
  KOKKOS_FUNCTION operator ArborX::Box() const { return {}; }
};
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakeBoundingVolume) {}
KOKKOS_FUNCTION void expand(FakeBoundingVolume, PrimitivePointOrBox) {}

struct FakePredicateGeometry {};
KOKKOS_FUNCTION ArborX::Point returnCentroid(FakePredicateGeometry) { return {}; }
KOKKOS_FUNCTION bool intersects(FakePredicateGeometry, FakeBoundingVolume) { return true; }
KOKKOS_FUNCTION float distance(FakePredicateGeometry, FakeBoundingVolume) { return 0.f; }
// clang-format on

struct PoorManLambda
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate, int) const
  {}
};
} // namespace Test

// Compile-only
void check_bounding_volume_and_predicate_geometry_type_requirements()
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  using Tree = ArborX::BasicBoundingVolumeHierarchy<MemorySpace,
                                                    Test::FakeBoundingVolume>;

  Kokkos::View<Test::PrimitivePointOrBox *, MemorySpace> primitives(
      "primitives", 0);
  Tree tree(ExecutionSpace{}, primitives);

  using SpatialPredicate =
      decltype(ArborX::intersects(Test::FakePredicateGeometry{}));
  Kokkos::View<SpatialPredicate *, MemorySpace> spatial_predicates(
      "spatial_predicates", 0);
  tree.query(ExecutionSpace{}, spatial_predicates, Test::PoorManLambda{});
#ifndef __NVCC__
  tree.query(ExecutionSpace{}, spatial_predicates,
             KOKKOS_LAMBDA(SpatialPredicate, int){});
#endif

  using NearestPredicate =
      decltype(ArborX::nearest(Test::FakePredicateGeometry{}));
  Kokkos::View<NearestPredicate *, MemorySpace> nearest_predicates(
      "nearest_predicates", 0);
  tree.query(ExecutionSpace{}, nearest_predicates, Test::PoorManLambda{});
#ifndef __NVCC__
  tree.query(ExecutionSpace{}, nearest_predicates,
             KOKKOS_LAMBDA(NearestPredicate, int){});
#endif
}
