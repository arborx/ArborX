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

#ifndef ARBORX_HYPERBOX_HPP
#define ARBORX_HYPERBOX_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp> // isfinite, sqrt
#include <Kokkos_ReductionIdentity.hpp>

namespace ArborX::ExperimentalHyperGeometry
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
template <int DIM, class Coordinate = float>
struct Box
{
  KOKKOS_FUNCTION
  constexpr Box()
  {
    for (int d = 0; d < DIM; ++d)
    {
      _min_corner[d] =
          Details::KokkosExt::ArithmeticTraits::finite_max<Coordinate>::value;
      _max_corner[d] =
          Details::KokkosExt::ArithmeticTraits::finite_min<Coordinate>::value;
    }
  }

  KOKKOS_FUNCTION
  constexpr Box(Point<DIM, Coordinate> const &min_corner,
                Point<DIM, Coordinate> const &max_corner)
      : _min_corner(min_corner)
      , _max_corner(max_corner)
  {}

  KOKKOS_FUNCTION
  constexpr auto &minCorner() { return _min_corner; }

  KOKKOS_FUNCTION
  constexpr auto const &minCorner() const { return _min_corner; }

  KOKKOS_FUNCTION
  constexpr auto &maxCorner() { return _max_corner; }

  KOKKOS_FUNCTION
  constexpr auto const &maxCorner() const { return _max_corner; }

  Point<DIM, Coordinate> _min_corner;
  Point<DIM, Coordinate> _max_corner;

  template <typename OtherBox,
            std::enable_if_t<GeometryTraits::is_box<OtherBox>{}> * = nullptr>
  KOKKOS_FUNCTION auto &operator+=(OtherBox const &other)
  {
    using Details::KokkosExt::max;
    using Details::KokkosExt::min;

    for (int d = 0; d < DIM; ++d)
    {
      minCorner()[d] = min(minCorner()[d], other.minCorner()[d]);
      maxCorner()[d] = max(maxCorner()[d], other.maxCorner()[d]);
    }
    return *this;
  }

  template <typename Point,
            std::enable_if_t<GeometryTraits::is_point_v<Point>> * = nullptr>
  KOKKOS_FUNCTION auto &operator+=(Point const &point)
  {
    using Details::KokkosExt::max;
    using Details::KokkosExt::min;

    for (int d = 0; d < DIM; ++d)
    {
      minCorner()[d] = min(minCorner()[d], point[d]);
      maxCorner()[d] = max(maxCorner()[d], point[d]);
    }
    return *this;
  }
};

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  using type = BoxTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  using type = Coordinate;
};

template <int DIM, typename Coordinate>
struct Kokkos::reduction_identity<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  KOKKOS_FUNCTION static ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>
  sum()
  {
    return {};
  }
};

namespace ArborX::Details::Dispatch
{
using GeometryTraits::BoxTag;
using GeometryTraits::PointTag;

// equals box-box
template <typename Box>
struct equals<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &l, Box const &r)
  {
    return Details::equals(l.minCorner(), r.minCorner()) &&
           Details::equals(l.maxCorner(), r.maxCorner());
  }
};

// isValid box
template <typename Box>
struct isValid<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
      if (r_d <= 0 || !Kokkos::isfinite(r_d))
        return false;
    }
    return true;
  }
};

// distance point-box
template <typename Point, typename Box>
struct distance<PointTag, BoxTag, Point, Box>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
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

// distance box-box
template <typename Box1, typename Box2>
struct distance<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static auto apply(Box1 const &box_a, Box2 const &box_b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box1>;
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

// expand a box to include a point
template <typename Box, typename Point>
struct expand<BoxTag, PointTag, Box, Point>
{
  KOKKOS_FUNCTION static void apply(Box &box, Point const &point)
  {
    box += point;
  }
};

// expand a box to include a box
template <typename Box1, typename Box2>
struct expand<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static void apply(Box1 &box, Box2 const &other)
  {
    box += other;
  }
};

// check if two axis-aligned bounding boxes intersect
template <typename Box1, typename Box2>
struct intersects<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box1 const &box,
                                              Box2 const &other)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box1>;
    for (int d = 0; d < DIM; ++d)
      if (box.minCorner()[d] > other.maxCorner()[d] ||
          box.maxCorner()[d] < other.minCorner()[d])
        return false;
    return true;
  }
};

// check it a box intersects with a point
template <typename Point, typename Box>
struct intersects<PointTag, BoxTag, Point, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              Box const &other)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (point[d] > other.maxCorner()[d] || point[d] < other.minCorner()[d])
        return false;
    return true;
  }
};

template <typename Box>
struct centroid<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr auto apply(Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    auto c = box.minCorner();
    for (int d = 0; d < DIM; ++d)
      c[d] = (c[d] + box.maxCorner()[d]) / 2;
    return c;
  }
};

} // namespace ArborX::Details::Dispatch
#endif
