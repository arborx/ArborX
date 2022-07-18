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

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include <ArborX.hpp>
#include <ArborX_Ray.hpp>

#include <boost/test/unit_test.hpp>

#include <numeric> //iota
#include <vector>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on

template <typename DeviceType>
struct NearestBoxToRay
{
  Kokkos::View<ArborX::Experimental::Ray *, DeviceType> rays;
  int k;
};

template <typename DeviceType>
struct BoxesIntersectedByRay
{
  Kokkos::View<ArborX::Experimental::Ray *, DeviceType> rays;
};

namespace ArborX
{
template <typename DeviceType>
struct AccessTraits<NearestBoxToRay<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int
  size(NearestBoxToRay<DeviceType> const &nearest_boxes)
  {
    return nearest_boxes.rays.size();
  }
  static KOKKOS_FUNCTION auto
  get(NearestBoxToRay<DeviceType> const &nearest_boxes, int i)
  {
    return nearest(nearest_boxes.rays(i), nearest_boxes.k);
  }
};

template <typename DeviceType>
struct AccessTraits<BoxesIntersectedByRay<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int
  size(BoxesIntersectedByRay<DeviceType> const &nearest_boxes)
  {
    return nearest_boxes.rays.size();
  }
  static KOKKOS_FUNCTION auto
  get(BoxesIntersectedByRay<DeviceType> const &nearest_boxes, int i)
  {
    return intersects(nearest_boxes.rays(i));
  }
};

} // namespace ArborX

BOOST_AUTO_TEST_SUITE(RayTraversals)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_ray_box_nearest, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{
  using memory_space = typename DeviceType::memory_space;
  typename DeviceType::execution_space exec_space;

  std::vector<ArborX::Box> boxes;
  for (unsigned int i = 0; i < 10; ++i)
    boxes.emplace_back(ArborX::Point(i, i, i),
                       ArborX::Point(i + 1, i + 1, i + 1));
  Kokkos::View<ArborX::Box *, DeviceType> device_boxes("boxes", 10);
  Kokkos::deep_copy(exec_space, device_boxes,
                    Kokkos::View<ArborX::Box *, Kokkos::HostSpace>(
                        boxes.data(), boxes.size()));

  ArborX::BVH<memory_space> const tree(exec_space, device_boxes);

  ArborX::Experimental::Ray ray{ArborX::Point{0, 0, 0},
                                ArborX::Experimental::Vector{.15, .1, 0.}};
  Kokkos::View<ArborX::Experimental::Ray *, DeviceType> device_rays("rays", 1);
  Kokkos::deep_copy(exec_space, device_rays, ray);

  BOOST_TEST(intersects(
      ray, ArborX::Box{ArborX::Point{0, 0, 0}, ArborX::Point{1, 1, 1}}));

  NearestBoxToRay<DeviceType> predicates{device_rays, 1};

  ARBORX_TEST_QUERY_TREE(exec_space, tree, predicates,
                         make_reference_solution<int>({0}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_ray_box_intersection, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{
  using memory_space = typename DeviceType::memory_space;
  typename DeviceType::execution_space exec_space;

  std::vector<ArborX::Box> boxes;
  for (unsigned int i = 0; i < 10; ++i)
    boxes.emplace_back(ArborX::Point(i, i, i),
                       ArborX::Point(i + 1, i + 1, i + 1));
  Kokkos::View<ArborX::Box *, DeviceType> device_boxes("boxes", 10);
  Kokkos::deep_copy(exec_space, device_boxes,
                    Kokkos::View<ArborX::Box *, Kokkos::HostSpace>(
                        boxes.data(), boxes.size()));

  ArborX::BVH<memory_space> const tree(exec_space, device_boxes);

  ArborX::Experimental::Ray ray{ArborX::Point{0, 0, 0},
                                ArborX::Experimental::Vector{.1, .1, .1}};
  Kokkos::View<ArborX::Experimental::Ray *, DeviceType> device_rays("rays", 1);
  Kokkos::deep_copy(exec_space, device_rays, ray);

  BoxesIntersectedByRay<DeviceType> predicates{device_rays};

  ARBORX_TEST_QUERY_TREE(
      exec_space, tree, predicates,
      make_reference_solution<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 10}));
}
BOOST_AUTO_TEST_SUITE_END()

template <typename DeviceType>
struct InsertIntersections
{
  mutable int count[2];
  Kokkos::View<int *[2], DeviceType> _ordered_intersections;

  template <typename Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  int const primitive_index) const
  {
    auto const predicate_index = getData(predicate);
    _ordered_intersections(Kokkos::atomic_fetch_inc(&count[predicate_index]),
                           predicate_index) = primitive_index;
  }
};

template <typename DeviceType>
struct BoxesIntersectedByRayOrdered
{
  Kokkos::View<ArborX::Experimental::Ray *, DeviceType> rays;
};

template <typename DeviceType>
struct ArborX::AccessTraits<BoxesIntersectedByRayOrdered<DeviceType>,
                            ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int
  size(BoxesIntersectedByRayOrdered<DeviceType> const &nearest_boxes)
  {
    return nearest_boxes.rays.size();
  }
  static KOKKOS_FUNCTION auto
  get(BoxesIntersectedByRayOrdered<DeviceType> const &nearest_boxes, int i)
  {
    return attach(ordered_intersects(nearest_boxes.rays(i)), i);
  }
};

BOOST_AUTO_TEST_SUITE(RayTraversals)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_ray_box_intersection_new, DeviceType,
                              ARBORX_TEST_DEVICE_TYPES)
{
  using memory_space = typename DeviceType::memory_space;
  typename DeviceType::execution_space exec_space;

  std::vector<ArborX::Box> boxes;
  int const n = 10;
  for (unsigned int i = 0; i < n; ++i)
    boxes.emplace_back(ArborX::Point(i, i, i),
                       ArborX::Point(i + 1, i + 1, i + 1));
  auto device_boxes = ArborXTest::toView<DeviceType>(boxes, "boxes");

  ArborX::BVH<memory_space> const tree(exec_space, device_boxes);

  Kokkos::View<ArborX::Experimental::Ray *, Kokkos::HostSpace> host_rays("rays",
                                                                         2);
  host_rays(0) = ArborX::Experimental::Ray{
      ArborX::Point{0, 0, 0},
      ArborX::Experimental::Vector{1. / n, 1. / n, 1. / n}};
  host_rays(1) = ArborX::Experimental::Ray{
      ArborX::Point{n, n, n},
      ArborX::Experimental::Vector{-1. / n, -1. / n, -1. / n}};
  auto device_rays = Kokkos::create_mirror_view_and_copy(
      typename DeviceType::memory_space{}, host_rays);
  Kokkos::View<int *[2], DeviceType> device_ordered_intersections(
      "ordered_intersections", n);

  BoxesIntersectedByRayOrdered<DeviceType> predicates{device_rays};

  tree.query(
      exec_space, predicates,
      InsertIntersections<DeviceType>{{0, 0}, device_ordered_intersections});
  auto const host_ordered_intersections = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, device_ordered_intersections);
  for (unsigned int i = 0; i < n; ++i)
  {
    BOOST_TEST(host_ordered_intersections(i, 0) == i);
    BOOST_TEST(host_ordered_intersections(i, 1) == n - 1 - i);
  }
}

BOOST_AUTO_TEST_SUITE_END()
