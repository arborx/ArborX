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
#ifndef ARBORX_DETAILS_ALGORITHMS_HPP
#define ARBORX_DETAILS_ALGORITHMS_HPP

#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMathFunctions.hpp>    // isfinite
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp> // min, max
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Details
{

template <typename Point,
          std::enable_if_t<GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION constexpr bool equals(Point const &l, Point const &r)
{
  constexpr int DIM = GeometryTraits::dimension<Point>::value;
  for (int d = 0; d < DIM; ++d)
    if (l[d] != r[d])
      return false;
  return true;
}

template <typename Box,
          std::enable_if_t<GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION constexpr bool equals(Box const &l, Box const &r)
{
  return equals(l.minCorner(), r.minCorner()) &&
         equals(l.maxCorner(), r.maxCorner());
}

template <typename Sphere,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION constexpr bool equals(Sphere const &l, Sphere const &r)
{
  return equals(l.centroid(), r.centroid()) && l.radius() == r.radius();
}

template <typename Point,
          std::enable_if_t<GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION bool isValid(Point const &p)
{
  using KokkosExt::isfinite;
  constexpr int DIM = GeometryTraits::dimension<Point>::value;
  for (int d = 0; d < DIM; ++d)
    if (!isfinite(p[d]))
      return false;
  return true;
}

template <typename Box,
          std::enable_if_t<GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION bool isValid(Box const &b)
{
  using KokkosExt::isfinite;
  constexpr int DIM = GeometryTraits::dimension<Box>::value;
  for (int d = 0; d < DIM; ++d)
  {
    auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
    if (r_d <= 0 || !isfinite(r_d))
      return false;
  }
  return true;
}

template <typename Sphere,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION bool isValid(Sphere const &s)
{
  using KokkosExt::isfinite;
  return isValid(s.centroid()) && isfinite(s.radius()) && (s.radius() >= 0.);
}

// distance point-point
template <typename Point,
          std::enable_if_t<GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION float distance(Point const &a, Point const &b)
{
  constexpr int DIM = GeometryTraits::dimension<Point>::value;
  float distance_squared = 0.0;
  for (int d = 0; d < DIM; ++d)
  {
    float tmp = b[d] - a[d];
    distance_squared += tmp * tmp;
  }
  return std::sqrt(distance_squared);
}

// distance point-box
template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point<Point>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION float distance(Point const &point, Box const &box)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");
  constexpr int DIM = GeometryTraits::dimension<Point>::value;

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
  return distance(point, projected_point);
}

// distance point-sphere
template <typename Point, typename Sphere,
          std::enable_if_t<GeometryTraits::is_point<Point>{} &&
                           GeometryTraits::is_sphere<Sphere>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION float distance(Point const &point, Sphere const &sphere)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Sphere>::value,
                "");
  using KokkosExt::max;
  return max(distance(point, sphere.centroid()) - sphere.radius(), 0.f);
}

// distance box-box
template <typename Box,
          std::enable_if_t<GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION float distance(Box const &box_a, Box const &box_b)
{
  constexpr int DIM = GeometryTraits::dimension<Box>::value;
  float distance_squared = 0.;
  for (int d = 0; d < DIM; ++d)
  {
    auto const a_min = box_a.minCorner()[d];
    auto const a_max = box_a.maxCorner()[d];
    auto const b_min = box_b.minCorner()[d];
    auto const b_max = box_b.maxCorner()[d];
    if (a_min > b_max)
    {
      float const delta = a_min - b_max;
      distance_squared += delta * delta;
    }
    else if (b_min > a_max)
    {
      float const delta = b_min - a_max;
      distance_squared += delta * delta;
    }
    else
    {
      // The boxes overlap on this axis: distance along this axis is zero.
    }
  }
  return std::sqrt(distance_squared);
}

// distance box-sphere
template <typename Sphere, typename Box,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION float distance(Sphere const &sphere, Box const &box)
{
  static_assert(GeometryTraits::dimension<Sphere>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");

  using KokkosExt::max;

  float distance_center_box = distance(sphere.centroid(), box);
  return max(distance_center_box - sphere.radius(), 0.f);
}

// expand an axis-aligned bounding box to include a point
template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point<Point>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void expand(Box &box, Point const &point)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");
  box += point;
}

// expand an axis-aligned bounding box to include another box
// NOTE: Box type is templated here to be able to use expand(box, box) in a
// Kokkos::parallel_reduce() in which case the arguments must be declared
// volatile.
template <typename BOX,
          std::enable_if_t<GeometryTraits::is_box<
              typename std::remove_volatile<BOX>::type>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void expand(BOX &box, BOX const &other)
{
  box += other;
}

// expand an axis-aligned bounding box to include a sphere
template <typename Box, typename Sphere,
          std::enable_if_t<GeometryTraits::is_box<Box>{} &&
                           GeometryTraits::is_sphere<Sphere>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void expand(Box &box, Sphere const &sphere)
{
  static_assert(GeometryTraits::dimension<Sphere>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");

  constexpr int DIM = GeometryTraits::dimension<Box>::value;

  using KokkosExt::max;
  using KokkosExt::min;
  for (int d = 0; d < DIM; ++d)
  {
    box.minCorner()[d] =
        min(box.minCorner()[d], sphere.centroid()[d] - sphere.radius());
    box.maxCorner()[d] =
        max(box.maxCorner()[d], sphere.centroid()[d] + sphere.radius());
  }
}

// check if two axis-aligned bounding boxes intersect
template <typename Box,
          std::enable_if_t<GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION constexpr bool intersects(Box const &box,
                                                 Box const &other)
{
  constexpr int DIM = GeometryTraits::dimension<Box>::value;
  for (int d = 0; d < DIM; ++d)
    if (box.minCorner()[d] > other.maxCorner()[d] ||
        box.maxCorner()[d] < other.minCorner()[d])
      return false;
  return true;
}

template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point<Point>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION constexpr bool intersects(Point const &point,
                                                 Box const &other)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");
  constexpr int DIM = GeometryTraits::dimension<Point>::value;
  for (int d = 0; d < DIM; ++d)
    if (point[d] > other.maxCorner()[d] || point[d] < other.minCorner()[d])
      return false;
  return true;
}

// check if a sphere intersects with an  axis-aligned bounding box
template <typename Sphere, typename Box,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION bool intersects(Sphere const &sphere, Box const &box)
{
  static_assert(GeometryTraits::dimension<Sphere>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");
  return distance(sphere.centroid(), box) <= sphere.radius();
}

template <typename Sphere, typename Point,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{} &&
                           GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION bool intersects(Sphere const &sphere, Point const &point)
{
  static_assert(GeometryTraits::dimension<Sphere>::value ==
                    GeometryTraits::dimension<Point>::value,
                "");
  return distance(sphere.centroid(), point) <= sphere.radius();
}

template <typename Sphere, typename Point,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{} &&
                           GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION bool intersects(Point const &point, Sphere const &sphere)
{
  static_assert(GeometryTraits::dimension<Sphere>::value ==
                    GeometryTraits::dimension<Point>::value,
                "");
  return intersects(sphere, point);
}

// calculate the centroid of a box
template <typename Box, typename Point,
          std::enable_if_t<GeometryTraits::is_box<Box>{} &&
                           GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void centroid(Box const &box, Point &c)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");
  constexpr int DIM = GeometryTraits::dimension<Point>::value;
  for (int d = 0; d < DIM; ++d)
    c[d] = (box.minCorner()[d] + box.maxCorner()[d]) / 2;
}

template <typename Point,
          std::enable_if_t<GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void centroid(Point const &point, Point &c)
{
  c = point;
}

template <typename Sphere, typename Point,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{} &&
                           GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void centroid(Sphere const &sphere, Point &c)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Sphere>::value,
                "");
  c = sphere.centroid();
}

template <typename Point,
          std::enable_if_t<GeometryTraits::is_point<Point>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION auto returnCentroid(Point const &point)
{
  return point;
}

template <typename Box,
          std::enable_if_t<GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION auto returnCentroid(Box const &box)
{
  constexpr int DIM = GeometryTraits::dimension<Box>::value;
  auto c = box.minCorner();
  for (int d = 0; d < DIM; ++d)
    c[d] = (c[d] + box.maxCorner()[d]) / 2;
  return c;
}

template <typename Sphere,
          std::enable_if_t<GeometryTraits::is_sphere<Sphere>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION auto returnCentroid(Sphere const &sphere)
{
  return sphere.centroid();
}

// transformation that maps the unit cube into a new axis-aligned box
// NOTE safe to perform in-place
template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point<Point>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_INLINE_FUNCTION void translateAndScale(Point const &in, Point &out,
                                              Box const &ref)
{
  static_assert(GeometryTraits::dimension<Point>::value ==
                    GeometryTraits::dimension<Box>::value,
                "");
  constexpr int DIM = GeometryTraits::dimension<Point>::value;
  for (int d = 0; d < DIM; ++d)
  {
    auto const a = ref.minCorner()[d];
    auto const b = ref.maxCorner()[d];
    out[d] = (a != b ? (in[d] - a) / (b - a) : 0);
  }
}

} // namespace Details
} // namespace ArborX

#endif
