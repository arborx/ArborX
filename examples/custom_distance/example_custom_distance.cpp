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
#include <vector>

// Define custom type that will be used for distance calculations
struct CustomPoint
{
  float x;
  float y;

  KOKKOS_FUNCTION auto operator[](int i) const
  {
    KOKKOS_ASSERT(i == 0 || i == 1);
    return (i == 0 ? x : y);
  }
  KOKKOS_FUNCTION auto &operator[](int i)
  {
    KOKKOS_ASSERT(i == 0 || i == 1);
    return (i == 0 ? x : y);
  }
};

template <>
struct ArborX::GeometryTraits::dimension<CustomPoint>
{
  static constexpr int value = 2;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<CustomPoint>
{
  using type = float;
};
template <>
struct ArborX::GeometryTraits::tag<CustomPoint>
{
  using type = ArborX::GeometryTraits::PointTag;
};

// Provide the required centroid function for the custom data type
KOKKOS_FUNCTION auto returnCentroid(CustomPoint const &point) { return point; }

// Provide the distance function between indexable getter geometry and the
// custom data type
KOKKOS_FUNCTION auto distance(CustomPoint const &p, CustomPoint const &q)
{
  return Kokkos::abs(p.x - q.x) * 2 + Kokkos::abs(p.y - q.y) / 2;
}

// Provide the distance function between the bounding volume geometry and
// the custom data type
using BoundingVolume = ArborX::Box<2>;
KOKKOS_FUNCTION auto distance(CustomPoint const &point,
                              BoundingVolume const &box)
{
  CustomPoint projected_point{
      Kokkos::clamp(point.x, box.minCorner()[0], box.maxCorner()[0]),
      Kokkos::clamp(point.y, box.minCorner()[0], box.maxCorner()[0])};
  return distance(point, projected_point);
}

// Callback to store the result indices
struct ExtractIndex
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &, Value const &value,
                                  Output const &out) const
  {
    out(value.index);
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  ExecutionSpace space;

  // In L2 metric, two bottom points are the closest.
  // In custom metric, left two points are the closest.
  //
  // | 1
  // |
  // | x
  // | 2 0
  // +-----------
  Kokkos::View<CustomPoint *, MemorySpace> points("Example::points", 3);
  auto points_host = Kokkos::create_mirror_view(points);
  points_host[0] = {1, 0};
  points_host[1] = {0, 3};
  points_host[2] = {0, 0};
  Kokkos::deep_copy(points, points_host);

  Kokkos::View<CustomPoint *, MemorySpace> query_points("Example::query_points",
                                                        1);
  auto query_points_host = Kokkos::create_mirror_view(query_points);
  query_points_host[0] = {0, 1};
  Kokkos::deep_copy(query_points, query_points_host);

  ArborX::BoundingVolumeHierarchy bvh(
      space, ArborX::Experimental::attach_indices(points));

  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  Kokkos::View<int *, MemorySpace> indices("Example::indices", 0);
  bvh.query(space, ArborX::Experimental::make_nearest(query_points, 2),
            ExtractIndex{}, indices, offsets);

  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  // Expected output:
  // offsets: 0 2
  // indices: 2 1
  std::cout << "offsets: ";
  std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\nindices: ";
  std::copy(indices_host.data(), indices_host.data() + indices.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";

  return 0;
}
