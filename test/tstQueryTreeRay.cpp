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

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include <ArborX.hpp>
#include <ArborXTest_LegacyTree.hpp>
#include <ArborX_Ray.hpp>
#include <misc/ArborX_Vector.hpp>

#include <boost/test/unit_test.hpp>

#include <numeric> //iota
#include <vector>

#include "Search_UnitTestHelpers.hpp"

#define ARBORX_TEST_DEVICE_TYPES                                               \
  std::tuple<Kokkos::DefaultExecutionSpace::device_type>

BOOST_AUTO_TEST_SUITE(RayTraversals)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_ray_box_nearest, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  typename DeviceType::execution_space exec_space;

  using Ray = ArborX::Experimental::Ray<>;
  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;

  using Tree =
      LegacyTree<ArborX::BoundingVolumeHierarchy<MemorySpace,
                                                 ArborX::PairValueIndex<Box>>>;

  std::vector<Box> boxes;
  for (unsigned int i = 0; i < 10; ++i)
    boxes.emplace_back(Point{(float)i, (float)i, (float)i},
                       Point{(float)i + 1, (float)i + 1, (float)i + 1});

  auto const tree = make<Tree>(exec_space, boxes);

  Ray ray{{0, 0, 0}, {.15f, .1f, 0.f}};
  Kokkos::View<Ray *, DeviceType> device_rays("rays", 1);
  Kokkos::deep_copy(exec_space, device_rays, ray);

  BOOST_TEST(intersects(ray, Box{{0, 0, 0}, {1, 1, 1}}));

  ARBORX_TEST_QUERY_TREE(exec_space, tree,
                         ArborX::Experimental::make_nearest(device_rays, 1),
                         make_reference_solution<int>({0}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_ray_box_intersection, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  typename DeviceType::execution_space exec_space;

  using Ray = ArborX::Experimental::Ray<>;
  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;

  using Tree =
      LegacyTree<ArborX::BoundingVolumeHierarchy<MemorySpace,
                                                 ArborX::PairValueIndex<Box>>>;

  std::vector<Box> boxes;
  for (unsigned int i = 0; i < 10; ++i)
    boxes.emplace_back(Point{(float)i, (float)i, (float)i},
                       Point{(float)i + 1, (float)i + 1, (float)i + 1});

  auto const tree = make<Tree>(exec_space, boxes);

  Ray ray{{0, 0, 0}, {.1f, .1f, .1f}};
  Kokkos::View<Ray *, DeviceType> device_rays("rays", 1);
  Kokkos::deep_copy(exec_space, device_rays, ray);

  ARBORX_TEST_QUERY_TREE(
      exec_space, tree, ArborX::Experimental::make_intersects(device_rays),
      make_reference_solution<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 10}));
}
BOOST_AUTO_TEST_SUITE_END()

template <typename DeviceType>
struct InsertIntersections
{
  // With ROCm version 5.2, we need to use a Kokkos::View to avoid a compiler
  // bug. See https://github.com/arborx/ArborX/issues/835
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 2)
  Kokkos::View<int[2], DeviceType> count;
#else
  mutable int count[2];
#endif
  Kokkos::View<int *[2], DeviceType> _ordered_intersections;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value) const
  {
    int const primitive_index = value.index;
    auto const predicate_index = getData(predicate);
    _ordered_intersections(Kokkos::atomic_fetch_inc(&count[predicate_index]),
                           predicate_index) = primitive_index;
  }
};

BOOST_AUTO_TEST_SUITE(RayTraversals)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_ray_box_intersection_new, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  typename DeviceType::execution_space exec_space;

  using Ray = ArborX::Experimental::Ray<>;
  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;

  using Tree =
      LegacyTree<ArborX::BoundingVolumeHierarchy<MemorySpace,
                                                 ArborX::PairValueIndex<Box>>>;

  std::vector<Box> boxes;
  int const n = 10;
  for (unsigned int i = 0; i < n; ++i)
    boxes.emplace_back(Point{(float)i, (float)i, (float)i},
                       Point{(float)i + 1, (float)i + 1, (float)i + 1});

  auto const tree = make<Tree>(exec_space, boxes);

  Kokkos::View<Ray *, Kokkos::HostSpace> host_rays("rays", 2);
  host_rays(0) = Ray{{0, 0, 0}, {1.f / n, 1.f / n, 1.f / n}};
  host_rays(1) = Ray{{n, n, n}, {-1.f / n, -1.f / n, -1.f / n}};
  auto device_rays =
      Kokkos::create_mirror_view_and_copy(MemorySpace{}, host_rays);
  Kokkos::View<int *[2], DeviceType> device_ordered_intersections(
      "ordered_intersections", n);

  auto predicates = ArborX::Experimental::attach_indices(
      ArborX::Experimental::make_ordered_intersects(device_rays));

#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 2)
  Kokkos::View<int[2], DeviceType> count("count");
  tree.query(
      exec_space, predicates,
      InsertIntersections<DeviceType>{count, device_ordered_intersections});
#else
  tree.query(
      exec_space, predicates,
      InsertIntersections<DeviceType>{{0, 0}, device_ordered_intersections});
#endif

  auto const host_ordered_intersections = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, device_ordered_intersections);
  for (unsigned int i = 0; i < n; ++i)
  {
    BOOST_TEST(host_ordered_intersections(i, 0) == i);
    BOOST_TEST(host_ordered_intersections(i, 1) == n - 1 - i);
  }
}

template <typename DeviceType, typename Coordinate>
auto makeOrderedIntersectsQueries(
    std::vector<ArborX::Experimental::Ray<Coordinate>> const &rays)
{
  int const n = rays.size();
  Kokkos::View<decltype(ArborX::Experimental::ordered_intersects(
                   ArborX::Experimental::Ray<Coordinate>{})) *,
               DeviceType>
      queries("Testing::intersecting_with_box_predicates", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::Experimental::ordered_intersects(rays[i]);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree_ordered_spatial_predicate, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{

  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;
  using Tree = LegacyTree<ArborX::BoundingVolumeHierarchy<
      MemorySpace, ArborX::PairValueIndex<ArborX::Box<3>>>>;
  Tree tree;
  BOOST_TEST(tree.empty());
  using ArborX::Details::equals;
  BOOST_TEST(equals(static_cast<ArborX::Box<3>>(tree.bounds()), {}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeOrderedIntersectsQueries<DeviceType, float>({})),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeOrderedIntersectsQueries<DeviceType, float>({
                             {},
                             {},
                         })),
                         make_reference_solution<int>({}, {0, 0, 0}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(single_leaf_tree_ordered_spatial_predicate,
                              DeviceType, ARBORX_TEST_DEVICE_TYPES)
{

  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;
  using Box = ArborX::Box<3>;
  using Tree =
      LegacyTree<ArborX::BoundingVolumeHierarchy<MemorySpace,
                                                 ArborX::PairValueIndex<Box>>>;

  auto const tree =
      make<Tree, Box>(ExecutionSpace{}, {
                                            {{{0., 0., 0.}}, {{1., 1., 1.}}},
                                        });

  BOOST_TEST(tree.size() == 1);
  using ArborX::Details::equals;
  BOOST_TEST(equals(static_cast<ArborX::Box<3>>(tree.bounds()),
                    {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeOrderedIntersectsQueries<DeviceType, float>({})),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeOrderedIntersectsQueries<DeviceType, float>({
                             {{0, 0, 0}, {1, 1, 1}},
                             {{4, 5, 6}, {7, 8, 9}},
                         })),
                         make_reference_solution<int>({0}, {0, 1, 1}));
}

BOOST_AUTO_TEST_SUITE_END()
