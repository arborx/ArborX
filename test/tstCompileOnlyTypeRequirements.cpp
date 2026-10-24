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

#include <ArborX_Box.hpp>
#include <ArborX_BruteForce.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Core.hpp>

#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp>
#endif

namespace Test
{
// NOTE only supporting Point and Box
using PrimitivePointOrBox = ArborX::Point<3>;
// using PrimitivePointOrBox = ArborX::Box;

// clang-format off
struct FakeBoundingVolume
{
};
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakeBoundingVolume) {}
KOKKOS_FUNCTION void expand(FakeBoundingVolume, PrimitivePointOrBox) {}
template<int DIM>
KOKKOS_FUNCTION void expand(ArborX::Box<DIM> &, FakeBoundingVolume) { }

struct FakePredicateGeometry {};
KOKKOS_FUNCTION ArborX::Point<3> returnCentroid(FakePredicateGeometry) { return {}; }
KOKKOS_FUNCTION bool intersects(FakePredicateGeometry, FakeBoundingVolume) { return true; }
KOKKOS_FUNCTION float distance(FakePredicateGeometry, FakeBoundingVolume) { return 0.f; }
KOKKOS_FUNCTION bool intersects(FakePredicateGeometry, PrimitivePointOrBox) { return true; }
KOKKOS_FUNCTION float distance(FakePredicateGeometry, PrimitivePointOrBox) { return 0.f; }
// clang-format on

struct PoorManLambda
{
  template <class Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate, Value) const
  {}
};
} // namespace Test

template <>
struct ArborX::GeometryTraits::dimension<Test::FakeBoundingVolume>
{
  static constexpr int value = 3;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<Test::FakeBoundingVolume>
{
  using type = float;
};

// Compile-only
void check_bounding_volume_and_predicate_geometry_type_requirements()
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  using Tree = ArborX::BoundingVolumeHierarchy<
      MemorySpace, Test::PrimitivePointOrBox,
      ArborX::Experimental::DefaultIndexableGetter, Test::FakeBoundingVolume>;

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
             KOKKOS_LAMBDA(SpatialPredicate, auto){});
#endif

  using NearestPredicate =
      decltype(ArborX::nearest(Test::FakePredicateGeometry{}));
  Kokkos::View<NearestPredicate *, MemorySpace> nearest_predicates(
      "nearest_predicates", 0);
  tree.query(ExecutionSpace{}, nearest_predicates, Test::PoorManLambda{});
#ifndef __NVCC__
  tree.query(ExecutionSpace{}, nearest_predicates,
             KOKKOS_LAMBDA(NearestPredicate, auto){});
#endif
}

namespace Test
{

// clang-format off
struct FakePrimitiveGeometry {};

KOKKOS_FUNCTION void expand(ArborX::Box<3> &, FakePrimitiveGeometry) {}
KOKKOS_FUNCTION ArborX::Point<3> returnCentroid(FakePrimitiveGeometry) { return {}; }
// clang-format on

} // namespace Test

template <>
struct ArborX::GeometryTraits::dimension<Test::FakePrimitiveGeometry>
{
  static constexpr int value = 3;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<Test::FakePrimitiveGeometry>
{
  using type = float;
};

// Compile-only
void check_hierarchy_for_custom_types()
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::View<Test::FakePrimitiveGeometry *, MemorySpace> primitives(
      "primitives", 0);
  ArborX::BoundingVolumeHierarchy tree(ExecutionSpace{}, primitives);
}

namespace Test
{

struct NonDefaultConstructibleValue
{
  ArborX::Point<3> point;
  int id;

  KOKKOS_FUNCTION NonDefaultConstructibleValue(ArborX::Point<3> p, int i)
      : point(p)
      , id(i)
  {}
  NonDefaultConstructibleValue() = delete;
};

struct NonDefaultConstructibleValueIndexableGetter
{
  KOKKOS_FUNCTION ArborX::Point<3>
  operator()(NonDefaultConstructibleValue const &value) const
  {
    return value.point;
  }
};

} // namespace Test

struct DummyCallback
{
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &, Value const &value,
                                  OutputFunctor const &out) const
  {
    out(value);
  }
};

// Compile-only
template <typename Tree, typename... Args>
void check_query_with_values_that_are_not_default_constructible_tmpl(
    Tree, Args &&...args)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  ExecutionSpace space{};

  using Value = Test::NonDefaultConstructibleValue;
  static_assert(!std::is_default_constructible_v<Value>);

  using IntersectsPredicate = decltype(ArborX::intersects(ArborX::Box<3>{}));
  using NearestPredicate = decltype(ArborX::nearest(ArborX::Box<3>{}, 1));

  Kokkos::View<Value *, MemorySpace> primitives(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primitives"), 0);
  Kokkos::View<IntersectsPredicate *, MemorySpace> intersects_predicates(
      "predicates", 0);
  Kokkos::View<NearestPredicate *, MemorySpace> nearest_predicates("predicates",
                                                                   0);
  Kokkos::View<Value *, MemorySpace> values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "values"), 0);
  Kokkos::View<int *, MemorySpace> offsets("offsets", 0);

  Tree tree(args..., space, primitives);
  tree.query(space, intersects_predicates, values, offsets);
  // For distributed tree, this will use the 2-phase query
  tree.query(space, nearest_predicates, values, offsets);
  tree.query(space, intersects_predicates, DummyCallback{}, values, offsets);
  // For distributed tree, this will use the 3-phase query
  tree.query(space, nearest_predicates, DummyCallback{}, values, offsets);
}

void check_query_with_values_that_are_not_default_constructible()
{
  check_query_with_values_that_are_not_default_constructible_tmpl(
      ArborX::BoundingVolumeHierarchy<
          typename Kokkos::DefaultExecutionSpace::memory_space,
          Test::NonDefaultConstructibleValue,
          Test::NonDefaultConstructibleValueIndexableGetter>());

  check_query_with_values_that_are_not_default_constructible_tmpl(
      ArborX::BruteForce<typename Kokkos::DefaultExecutionSpace::memory_space,
                         Test::NonDefaultConstructibleValue,
                         Test::NonDefaultConstructibleValueIndexableGetter>());
#ifdef ARBORX_ENABLE_MPI
  check_query_with_values_that_are_not_default_constructible_tmpl(
      ArborX::DistributedTree<
          typename Kokkos::DefaultExecutionSpace::memory_space,
          Test::NonDefaultConstructibleValue,
          Test::NonDefaultConstructibleValueIndexableGetter>(),
      MPI_COMM_WORLD);
#endif
}
