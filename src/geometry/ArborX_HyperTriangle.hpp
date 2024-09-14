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
#ifndef ARBORX_TRIANGLE_HPP
#define ARBORX_TRIANGLE_HPP

#include <ArborX_Point.hpp>

namespace ArborX::ExperimentalHyperGeometry
{
// need to add a protection that
// the points are not on the same line.
template <int DIM, class Coordinate = float>
struct Triangle
{
  Point<DIM, Coordinate> a;
  Point<DIM, Coordinate> b;
  Point<DIM, Coordinate> c;
};

template <int DIM, class Coordinate>
Triangle(Point<DIM, Coordinate>, Point<DIM, Coordinate>, Point<DIM, Coordinate>)
    -> Triangle<DIM, Coordinate>;

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  using type = TriangleTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
