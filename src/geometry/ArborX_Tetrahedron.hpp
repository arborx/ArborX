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

#ifndef ARBORX_TETRAHEDRON_HPP
#define ARBORX_TETRAHEDRON_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <detail/ArborX_GeometryAlgorithms.hpp>

#include <Kokkos_Array.hpp>

namespace ArborX::ExperimentalHyperGeometry
{
// Need to add a protection that the points are not in the same plane
template <class Coordinate = float>
struct Tetrahedron
{
  Point<3, Coordinate> a;
  Point<3, Coordinate> b;
  Point<3, Coordinate> c;
  Point<3, Coordinate> d;
};

template <class Coordinate>
Tetrahedron(Point<3, Coordinate>, Point<3, Coordinate>, Point<3, Coordinate>,
            Point<3, Coordinate>) -> Tetrahedron<Coordinate>;

} // namespace ArborX::ExperimentalHyperGeometry

template <class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>>
{
  static constexpr int value = 3;
};
template <class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>>
{
  using type = TetrahedronTag;
};
template <class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>>
{
  using type = Coordinate;
};

namespace ArborX::Details::Dispatch
{
using GeometryTraits::BoxTag;
using GeometryTraits::PointTag;
using GeometryTraits::TetrahedronTag;

// expand a box to include a tetrahedron
template <typename Box, typename Tetrahedron>
struct expand<BoxTag, TetrahedronTag, Box, Tetrahedron>
{
  KOKKOS_FUNCTION static void apply(Box &box, Tetrahedron const &tet)
  {
    Details::expand(box, tet.a);
    Details::expand(box, tet.b);
    Details::expand(box, tet.c);
    Details::expand(box, tet.d);
  }
};

template <typename Point, typename Tetrahedron>
struct intersects<PointTag, TetrahedronTag, Point, Tetrahedron>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              Tetrahedron const &tet)
  {
    static_assert(GeometryTraits::dimension_v<Point> == 3);

    constexpr int N = 4;
    Kokkos::Array<decltype(tet.a), N> v = {tet.a, tet.b, tet.c, tet.d};

    // For every plane check that the vertex lies within the same halfspace as
    // the other tetrahedron vertex. This is a simple but possibly not very
    // efficient algorithm.
    for (int j = 0; j < N; ++j)
    {
      auto normal = (v[(j + 1) % N] - v[j]).cross(v[(j + 2) % N] - v[j]);

      bool same_half_space =
          (normal.dot(v[(j + 3) % N] - v[j]) * normal.dot(point - v[j]) >= 0);
      if (!same_half_space)
        return false;
    }
    return true;
  }
};

} // namespace ArborX::Details::Dispatch

#endif
