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
#ifndef ARBORX_DETAILS_GEOMETRY_DISTANCE_HPP
#define ARBORX_DETAILS_GEOMETRY_DISTANCE_HPP

#include "ArborX_Equals.hpp"
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Triangle.hpp>
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Clamp.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_MinMax.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct distance;

}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION auto distance(Geometry1 const &geometry1,
                                     Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::distance<GeometryTraits::tag_t<Geometry1>,
                            GeometryTraits::tag_t<Geometry2>, Geometry1,
                            Geometry2>::apply(geometry1, geometry2);
}

namespace Dispatch
{

using namespace GeometryTraits;

// distance point-point
template <typename Point1, typename Point2>
struct distance<PointTag, PointTag, Point1, Point2>
{
  KOKKOS_FUNCTION static auto apply(Point1 const &a, Point2 const &b)
  {
    constexpr int DIM = dimension_v<Point1>;
    // Points may have different coordinate types. Try using implicit
    // conversion to get the best one.
    using Coordinate = decltype(b[0] - a[0]);
    Coordinate distance_squared = 0;
    for (int d = 0; d < DIM; ++d)
    {
      auto tmp = b[d] - a[d];
      distance_squared += tmp * tmp;
    }
    return Kokkos::sqrt(distance_squared);
  }
};

// distance point-box
template <typename Point, typename Box>
struct distance<PointTag, BoxTag, Point, Box>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Box const &box)
  {
    constexpr int DIM = dimension_v<Point>;
    Point projected_point;
    for (int d = 0; d < DIM; ++d)
    {
      if (point[d] < box.minCorner()[d])
        projected_point[d] = box.minCorner()[d];
      else if (point[d] > box.maxCorner()[d])
        projected_point[d] = box.maxCorner()[d];
      else
        projected_point[d] = point[d];
    }
    return Details::distance(point, projected_point);
  }
};

template <typename Box, typename Point>
struct distance<BoxTag, PointTag, Box, Point>
{
  KOKKOS_FUNCTION static auto apply(Box const &box, Point const &point)
  {
    return Details::distance(point, box);
  }
};

// distance point-sphere
template <typename Point, typename Sphere>
struct distance<PointTag, SphereTag, Point, Sphere>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Sphere const &sphere)
  {
    using Kokkos::max;
    using Coordinate = GeometryTraits::coordinate_type_t<Sphere>;
    return max(Details::distance(point, sphere.centroid()) - sphere.radius(),
               (Coordinate)0);
  }
};

// distance sphere-point
template <typename Sphere, typename Point>
struct distance<SphereTag, PointTag, Sphere, Point>
{
  KOKKOS_FUNCTION static auto apply(Sphere const &sphere, Point const &point)
  {
    return Details::distance(point, sphere);
  }
};

// distance point-triangle
template <typename Point, typename Triangle>
struct distance<PointTag, TriangleTag, Point, Triangle>
{
  static constexpr int DIM = dimension_v<Point>;
  using Coordinate = coordinate_type_t<Triangle>;

  static_assert(DIM == 2 || DIM == 3);

  KOKKOS_FUNCTION static auto combine(Point const &a, Point const &b,
                                      Point const &c, Coordinate u,
                                      Coordinate v, Coordinate w)
  {
    Point r;
    for (int d = 0; d < DIM; ++d)
      r[d] = u * a[d] + v * b[d] + w * c[d];
    return r;
  }
  KOKKOS_FUNCTION static auto closest_point(Point const &p, Point const &a,
                                            Point const &b, Point const &c)
  {
    /* Zones
           \ 2/
            \/
        5   /\b  6
           /  \
          /    \
      \  /   0  \  /
       \/a______c\/
      1 |    4   | 3
        |        |
    */

    auto const ab = b - a;
    auto const ac = c - a;
    auto const ap = p - a;

    auto const d1 = ab.dot(ap);
    auto const d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) // zone 1
      return a;

    auto const bp = p - b;
    auto const d3 = ab.dot(bp);
    auto const d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) // zone 2
      return b;

    auto const cp = p - c;
    auto const d5 = ab.dot(cp);
    auto const d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) // zone 3
      return c;

    auto const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) // zone 5
    {
      auto const v = d1 / (d1 - d3);
      return combine(a, b, c, 1 - v, v, 0);
    }

    auto const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) // zone 4
    {
      auto const v = d2 / (d2 - d6);
      return combine(a, b, c, 1 - v, 0, v);
    }

    auto const va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) // zone 6
    {
      auto const v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return combine(a, b, c, 0, 1 - v, v);
    }

    // zone 0
    auto const denom = 1 / (va + vb + vc);
    auto const v = vb * denom;
    auto const w = vc * denom;
    return combine(a, b, c, 1 - v - w, v, w);
  }

  KOKKOS_FUNCTION static auto apply(Point const &p, Triangle const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    return Details::distance(p, closest_point(p, a, b, c));
  }
};

// distance point-tetrahedron
template <typename Point, typename Tetrahedron>
struct distance<PointTag, TetrahedronTag, Point, Tetrahedron>
{
  static constexpr int DIM = dimension_v<Point>;
  using Coordinate = coordinate_type_t<Tetrahedron>;

  KOKKOS_FUNCTION static auto apply(Point const &point, Tetrahedron const &tet)
  {
    static_assert(GeometryTraits::dimension_v<Point> == 3);

    constexpr int N = 4;
    Kokkos::Array<decltype(tet.a), N> v = {tet.a, tet.b, tet.c, tet.d};

    // For every plane check that the vertex lies within the same halfspace as
    // the other tetrahedron vertex (similar to the current intersects
    // algorithm). If not, compute the distance to the corresponding triangle.
    constexpr auto fmax =
        Details::KokkosExt::ArithmeticTraits::finite_max<Coordinate>::value;
    auto min_distance = fmax;
    for (int j = 0; j < N; ++j)
    {
      auto normal = (v[(j + 1) % N] - v[j]).cross(v[(j + 2) % N] - v[j]);

      bool same_half_space =
          (normal.dot(v[(j + 3) % N] - v[j]) * normal.dot(point - v[j]) >= 0);
      if (!same_half_space)
        min_distance =
            Kokkos::min(min_distance,
                        Details::distance(point, Triangle{v[j], v[(j + 1) % N],
                                                          v[(j + 2) % N]}));
    }
    return (min_distance != fmax ? min_distance : static_cast<Coordinate>(0));
  }
};

// distance tetrahedron-point
template <typename Tetrahedron, typename Point>
struct distance<TetrahedronTag, PointTag, Tetrahedron, Point>
{
  static constexpr int DIM = dimension_v<Point>;
  using Coordinate = coordinate_type_t<Tetrahedron>;

  KOKKOS_FUNCTION static auto apply(Tetrahedron const &tet, Point const &p)
  {
    return Details::distance(p, tet);
  }
};

// distance box-box
template <typename Box1, typename Box2>
struct distance<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static auto apply(Box1 const &box_a, Box2 const &box_b)
  {
    constexpr int DIM = dimension_v<Box1>;
    // Boxes may have different coordinate types. Try using implicit
    // conversion to get the best one.
    using Coordinate = decltype(box_b.minCorner()[0] - box_a.minCorner()[0]);
    Coordinate distance_squared = 0;
    for (int d = 0; d < DIM; ++d)
    {
      auto const a_min = box_a.minCorner()[d];
      auto const a_max = box_a.maxCorner()[d];
      auto const b_min = box_b.minCorner()[d];
      auto const b_max = box_b.maxCorner()[d];
      if (a_min > b_max)
      {
        auto const delta = a_min - b_max;
        distance_squared += delta * delta;
      }
      else if (b_min > a_max)
      {
        auto const delta = b_min - a_max;
        distance_squared += delta * delta;
      }
      else
      {
        // The boxes overlap on this axis: distance along this axis is zero.
      }
    }
    return Kokkos::sqrt(distance_squared);
  }
};

// distance sphere-box
template <typename Sphere, typename Box>
struct distance<SphereTag, BoxTag, Sphere, Box>
{
  KOKKOS_FUNCTION static auto apply(Sphere const &sphere, Box const &box)
  {
    using Kokkos::max;
    using Coordinate = GeometryTraits::coordinate_type_t<Sphere>;

    auto distance_center_box = Details::distance(sphere.centroid(), box);
    return max(distance_center_box - sphere.radius(), (Coordinate)0);
  }
};

// distance sphere-box
template <typename Box, typename Sphere>
struct distance<BoxTag, SphereTag, Box, Sphere>
{
  KOKKOS_FUNCTION static auto apply(Box const &box, Sphere const &sphere)
  {
    return Details::distance(sphere, box);
  }
};

template <typename Point, typename Segment>
struct distance<PointTag, SegmentTag, Point, Segment>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Segment const &segment)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;

    if (Details::equals(segment.a, segment.b))
      return Details::distance(point, segment.a);

    auto const dir = segment.b - segment.a;

    // The line of the segment [a,b] is parametrized as a + t * (b - a).
    // Find the projection of the point to that line, and clamp it.
    auto t =
        Kokkos::clamp(dir.dot(point - segment.a) / dir.dot(dir),
                      static_cast<Coordinate>(0), static_cast<Coordinate>(1));

    Point projection;
    for (int d = 0; d < DIM; ++d)
      projection[d] = segment.a[d] + t * dir[d];

    return Details::distance(point, projection);
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
