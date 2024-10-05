/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_GEOMETRY_CENTROID_HPP
#define ARBORX_DETAILS_GEOMETRY_CENTROID_HPP

#include "ArborX_GeometryExpand.hpp"
#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename Tag, typename Geometry>
struct centroid;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION decltype(auto) returnCentroid(Geometry const &geometry)
{
  return Dispatch::centroid<typename GeometryTraits::tag_t<Geometry>,
                            Geometry>::apply(geometry);
}

namespace Dispatch
{

using namespace GeometryTraits;

template <typename Point>
struct centroid<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr auto apply(Point const &point)
  {
    return point;
  }
};

template <typename Box>
struct centroid<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr auto apply(Box const &box)
  {
    constexpr int DIM = dimension_v<Box>;
    auto c = box.minCorner();
    for (int d = 0; d < DIM; ++d)
      c[d] = (c[d] + box.maxCorner()[d]) / 2;
    return c;
  }
};

template <typename Sphere>
struct centroid<SphereTag, Sphere>
{
  KOKKOS_FUNCTION static constexpr auto apply(Sphere const &sphere)
  {
    return sphere.centroid();
  }
};

template <typename Triangle>
struct centroid<TriangleTag, Triangle>
{
  KOKKOS_FUNCTION static constexpr auto apply(Triangle const &triangle)
  {
    constexpr int DIM = dimension_v<Triangle>;
    auto c = triangle.a;
    for (int d = 0; d < DIM; ++d)
      c[d] = (c[d] + triangle.b[d] + triangle.c[d]) / 3;
    return c;
  }
};

template <typename KDOP>
struct centroid<KDOPTag, KDOP>
{
  KOKKOS_FUNCTION static auto apply(KDOP const &kdop)
  {
    // FIXME approximation
    using Box = Box<dimension_v<KDOP>, coordinate_type_t<KDOP>>;
    Box box;
    Details::expand(box, kdop);
    return centroid<BoxTag, Box>::apply(box);
  }
};

template <typename Tetrahedron>
struct centroid<TetrahedronTag, Tetrahedron>
{
  KOKKOS_FUNCTION static constexpr auto apply(Tetrahedron const &tet)
  {
    constexpr int DIM = dimension_v<Tetrahedron>;
    static_assert(DIM == 3);
    auto c = tet.a;
    for (int d = 0; d < DIM; ++d)
      c[d] = (c[d] + tet.b[d] + tet.c[d] + tet.d[d]) / 4;
    return c;
  }
};

template <typename Segment>
struct centroid<SegmentTag, Segment>
{
  KOKKOS_FUNCTION static auto apply(Segment const &segment)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Segment>;
    using Coordinate = GeometryTraits::coordinate_type_t<Segment>;

    // WARNING implicit requirement on KDOP first DIM directions
    Point<DIM, Coordinate> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = (segment._start[d] + segment._end[d]) / 2;
    return point;
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
