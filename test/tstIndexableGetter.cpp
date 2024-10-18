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
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_LinearBVH.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Indexable.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

using namespace ArborX::Details;

#include <ArborX_Point.hpp>

template <typename MemorySpace>
struct PointCloud
{
  ArborX::Point<3> *data;
  int n;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<PointCloud<MemorySpace>, ArborX::PrimitivesTag>
{
  using Points = PointCloud<MemorySpace>;

  static KOKKOS_FUNCTION std::size_t size(Points const &points)
  {
    return points.n;
  }
  static KOKKOS_FUNCTION auto get(Points const &points, std::size_t i)
  {
    return points.data[i];
  }
  using memory_space = MemorySpace;
};

template <typename MemorySpace>
struct PairPointIndexCloud
{
  ArborX::Point<3> *data;
  int n;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<PairPointIndexCloud<MemorySpace>,
                            ArborX::PrimitivesTag>
{
  using Points = PairPointIndexCloud<MemorySpace>;

  static KOKKOS_FUNCTION std::size_t size(Points const &points)
  {
    return points.n;
  }
  static KOKKOS_FUNCTION auto get(Points const &points, std::size_t i)
  {
    return ArborX::PairValueIndex<ArborX::Point<3>, int>{points.data[i],
                                                         (int)i};
  }
  using memory_space = MemorySpace;
};

template <typename ExecutionSpace, typename Indexables, typename Box>
inline void calculateBoundingBoxOfTheScene(ExecutionSpace const &space,
                                           Indexables const &indexables,
                                           Box &scene_bounding_box)
{
  Kokkos::parallel_reduce(
      "ArborX::TreeConstruction::calculate_bounding_box_of_the_scene",
      Kokkos::RangePolicy(space, 0, indexables.size()),
      KOKKOS_LAMBDA(int i, Box &update) { expand(update, indexables(i)); },
      ArborX::Details::GeometryReducer<Box>{scene_bounding_box});
}

BOOST_AUTO_TEST_SUITE(IndexableGetterAccess)

BOOST_AUTO_TEST_CASE_TEMPLATE(indexables, DeviceType, ARBORX_DEVICE_TYPES)
{
  // Test that the two-level wrapping Data -> AccessValues -> Indexables by
  // using default Indexable works correctly.

  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  using ArborX::Details::equals;

  using Point = ArborX::Point<3>;

  Kokkos::View<Point *, MemorySpace> points("Testing::points", 2);
  auto points_host = Kokkos::create_mirror_view(points);
  points_host(0) = {-1, -1, -1};
  points_host(1) = {1, 1, 1};
  Kokkos::deep_copy(points, points_host);

  ArborX::Box scene_bounding_box{{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};

  {
    PointCloud<MemorySpace> points_cloud{points.data(), (int)points.size()};

    using Primitives = ArborX::Details::AccessValues<decltype(points_cloud),
                                                     ArborX::PrimitivesTag>;
    using IndexableGetter =
        ArborX::Experimental::Indexable<typename Primitives::value_type>;

    Primitives primitives(points_cloud);

    ArborX::Details::Indexables indexables{primitives, IndexableGetter{}};

    ArborX::Box<3> box;
    calculateBoundingBoxOfTheScene(ExecutionSpace{}, indexables, box);
    BOOST_TEST(equals(box, scene_bounding_box));
  }

  {
    PairPointIndexCloud<MemorySpace> points_cloud{points.data(),
                                                  (int)points.size()};

    using Primitives = ArborX::Details::AccessValues<decltype(points_cloud),
                                                     ArborX::PrimitivesTag>;
    using IndexableGetter =
        ArborX::Experimental::Indexable<typename Primitives::value_type>;
    Primitives primitives(points_cloud);

    ArborX::Details::Indexables indexables{primitives, IndexableGetter{}};

    ArborX::Box<3> box;
    calculateBoundingBoxOfTheScene(ExecutionSpace{}, indexables, box);
    BOOST_TEST(equals(box, scene_bounding_box));
  }
}

template <typename MemorySpace, typename Geometry>
struct ReturnByValueIndexableGetter
{
  Kokkos::View<Geometry *, MemorySpace> _geometries;

  KOKKOS_FUNCTION auto operator()(ArborX::PairValueIndex<Geometry> value) const
  {
    return _geometries(value.index);
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(indexables_by_value, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // Test that the there is no reference to temporary when indexable getter
  // returns by value. Would produce a warning.

  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  using Point = ArborX::Point<3>;

  Kokkos::View<Point *, MemorySpace> points("Testing::points", 2);
  auto points_host = Kokkos::create_mirror_view(points);
  points_host(0) = {-1, -1, -1};
  points_host(1) = {1, 1, 1};
  Kokkos::deep_copy(points, points_host);

  using Value = ArborX::PairValueIndex<Point>;

  using IndexableGetter = ReturnByValueIndexableGetter<MemorySpace, Point>;
  ArborX::BoundingVolumeHierarchy<MemorySpace, Value, IndexableGetter> bvh(
      ExecutionSpace{}, ArborX::Experimental::attach_indices(points),
      IndexableGetter{points});

  Kokkos::View<ArborX::Nearest<Point> *, MemorySpace> nearest_queries(
      "Testing::nearest_queries", 1);
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  nearest_queries_host(0) = ArborX::nearest<Point>({0, 0, 0}, 1);
  deep_copy(nearest_queries, nearest_queries_host);

  // We need to go through HappyTreeFriends to trigger the warning,
  // as that's where we use `getIndexable()`
  Kokkos::View<int *, MemorySpace> offsets("Testing::offsets", 0);
  Kokkos::View<Value *, MemorySpace> values("Testing::values", 0);
  bvh.query(ExecutionSpace{}, nearest_queries, values, offsets);
  BOOST_TEST(offsets.size() == 2); // to prevent compiler to optimize out
}

BOOST_AUTO_TEST_SUITE_END()
