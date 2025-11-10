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
#ifndef ARBORX_DETAILS_GEOMETRY_CLOSEST_POINT_HPP
#define ARBORX_DETAILS_GEOMETRY_CLOSEST_POINT_HPP

#include "ArborX_Equals.hpp"
#include <ArborX_GeometryTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Clamp.hpp>

namespace ArborX
{
namespace Details::Dispatch
{
template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct closestPoint;
}

namespace Experimental
{
template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION auto closestPoint(Geometry1 const &geometry1,
                                         Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Details::Dispatch::closestPoint<
      GeometryTraits::tag_t<Geometry1>, GeometryTraits::tag_t<Geometry2>,
      Geometry1, Geometry2>::apply(geometry1, geometry2);
}
} // namespace Experimental

namespace Details::Dispatch
{

using namespace GeometryTraits;

template <typename Point, typename Box>
struct closestPoint<PointTag, BoxTag, Point, Box>
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
    return projected_point;
  }
};

template <typename Point, typename Triangle>
struct closestPoint<PointTag, TriangleTag, Point, Triangle>
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

  KOKKOS_FUNCTION static auto apply(Point const &p, Triangle const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

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
};

template <typename Point, typename Tetrahedron>
struct closestPoint<PointTag, TetrahedronTag, Point, Tetrahedron>
{
  static constexpr int DIM = dimension_v<Point>;
  using Coordinate = coordinate_type_t<Tetrahedron>;

  KOKKOS_FUNCTION static auto apply(Point const &, Tetrahedron const &)
  {
    static_assert(GeometryTraits::dimension_v<Point> == 3);
    static_assert(DIM >= 0, "Not implemented yet");
  }
};

template <typename Point, typename Segment>
struct closestPoint<PointTag, SegmentTag, Point, Segment>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Segment const &segment)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;

    if (Details::equals(segment.a, segment.b))
      return segment.a;

    auto const dir = segment.b - segment.a;

    // The line of the segment [a,b] is parametrized as a + t * (b - a).
    // Find the projection of the point to that line, and clamp it.
    auto t =
        Kokkos::clamp(dir.dot(point - segment.a) / dir.dot(dir),
                      static_cast<Coordinate>(0), static_cast<Coordinate>(1));

    Point projection;
    for (int d = 0; d < DIM; ++d)
      projection[d] = segment.a[d] + t * dir[d];

    return projection;
  }
};

} // namespace Details::Dispatch

} // namespace ArborX

#endif
