/****************************************************************************
 * Copyright (c) 2017-2024 by the ArborX authors                            *
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

constexpr int DIM = 2;
using Coordinate = float;

using Point = ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>;
using Box = ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>;

// Define custom type that will be used for distance calculations
struct CustomPoint : public Point
{};

// Provide the required centroid function for the custom data type
KOKKOS_FUNCTION auto returnCentroid(CustomPoint const &point)
{
  return Kokkos::bit_cast<Point>(point);
}

// Provide the distance function between indexable getter geometry and the
// custom data type
KOKKOS_FUNCTION auto distance(CustomPoint const &custom_point,
                              Point const &point)
{
  static_assert(DIM == 2);
  return Kokkos::abs(custom_point[0] - point[0]) * 2 +
         Kokkos::abs(custom_point[1] - point[1]) / 2;
}

// Provide the distance function between the bounding volume geometry and
// the custom data type
KOKKOS_FUNCTION auto distance(CustomPoint const &point, Box const &box)
{
  static_assert(DIM == 2);
  Point projected_point;
  for (int d = 0; d < DIM; ++d)
    projected_point[d] =
        Kokkos::clamp(point[d], box.minCorner()[d], box.maxCorner()[d]);
  return distance(point, projected_point);
}

// Allow simple wrapping of regular points into the custom data
template <typename Points>
struct CustomPoints
{
  Points points;
};

template <typename Points>
struct ArborX::AccessTraits<CustomPoints<Points>, ArborX::PredicatesTag>
{
  using Self = CustomPoints<Points>;
  using memory_space = typename Points::memory_space;

  // Search for the two nearest neighbors
  static constexpr int k = 2;

  static KOKKOS_FUNCTION auto size(Self const &x) { return x.points.size(); }
  static KOKKOS_FUNCTION decltype(auto) get(Self const &x, int i)
  {
    return ArborX::nearest(Kokkos::bit_cast<CustomPoint>(x.points(i)), k);
  }
};

// Callback to store the resulting distances
struct DistanceCallback
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &value,
                                  Output const &out) const
  {
    out(distance(ArborX::getGeometry(query), value));
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
  // x
  //
  // o
  // x x
  Kokkos::View<Point *, MemorySpace> points("Example::points", 3);
  auto points_host = Kokkos::create_mirror_view(points);
  points_host[0] = {0, 0};
  points_host[1] = {1, 0};
  points_host[2] = {0, 3};
  Kokkos::deep_copy(points, points_host);

  Kokkos::View<Point *, MemorySpace> query_points("Example::query_points", 1);
  auto query_points_host = Kokkos::create_mirror_view(query_points);
  query_points_host[0] = {0, 1};
  Kokkos::deep_copy(query_points, query_points_host);

  ArborX::BVH<MemorySpace, Point> bvh(space, points);

  Kokkos::View<float *, MemorySpace> distances("Example::distances", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  bvh.query(space, CustomPoints<decltype(query_points)>{query_points},
            DistanceCallback{}, distances, offsets);

  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto distances_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, distances);

  // Expected output:
  // offsets: 0 2
  // distances: 0.5 1
  std::cout << "offsets: ";
  std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\ndistances: ";
  std::copy(distances_host.data(), distances_host.data() + distances.size(),
            std::ostream_iterator<float>(std::cout, " "));
  std::cout << "\n";

  return 0;
}
