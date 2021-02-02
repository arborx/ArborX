/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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
using FakePrimitive = ArborX::Point;
// using FakePrimitive = ArborX::Box;

// clang-format off
struct FakeBoundingVolume
{
  KOKKOS_FUNCTION FakeBoundingVolume &operator+=(FakePrimitive) { return *this; }
  KOKKOS_FUNCTION void operator+=(FakePrimitive) volatile {}
  KOKKOS_FUNCTION operator ArborX::Box() const { return {}; }
};
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakeBoundingVolume) {}
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakePrimitive) {}

struct FakePredicateGeometry {};
KOKKOS_FUNCTION ArborX::Point returnCentroid(FakePredicateGeometry) { return {}; }
KOKKOS_FUNCTION bool intersects(FakePredicateGeometry, FakeBoundingVolume) { return true; }
// clang-format on
} // namespace Test

// Compile-only
void check_bounding_volume_and_predicate_geometry_type_requirements()
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  using Tree = ArborX::BasicBoundingVolumeHierarchy<MemorySpace,
                                                    Test::FakeBoundingVolume>;

  Kokkos::View<Test::FakePrimitive *, MemorySpace> primitives("primitives", 0);
  Tree tree(ExecutionSpace{}, primitives);

  Kokkos::View<decltype(ArborX::intersects(Test::FakePredicateGeometry{})) *,
               MemorySpace>
      predicates("predicates", 0);
  tree.query(ExecutionSpace{}, predicates, [](auto, int) {});
}

int main() { return 0; }
