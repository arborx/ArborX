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

#include <ArborX_Box.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMathFunctions.hpp> // isfinite
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Details
{

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Point const &l, Point const &r)
{
  for (int d = 0; d < 3; ++d)
    if (l[d] != r[d])
      return false;
  return true;
}

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Box const &l, Box const &r)
{
  return equals(l.minCorner(), r.minCorner()) &&
         equals(l.maxCorner(), r.maxCorner());
}

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Sphere const &l, Sphere const &r)
{
  return equals(l.centroid(), r.centroid()) && l.radius() == r.radius();
}

KOKKOS_INLINE_FUNCTION
bool isValid(Point const &p)
{
  using KokkosExt::isfinite;
  for (int d = 0; d < 3; ++d)
    if (!isfinite(p[d]))
      return false;
  return true;
}

KOKKOS_INLINE_FUNCTION
bool isValid(Box const &b)
{
  using KokkosExt::isfinite;
  for (int d = 0; d < 3; ++d)
  {
    auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
    if (r_d <= 0 || !isfinite(r_d))
      return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool isValid(Sphere const &s)
{
  using KokkosExt::isfinite;
  return isValid(s.centroid()) && isfinite(s.radius()) && (s.radius() >= 0.);
}

// distance point-point
KOKKOS_INLINE_FUNCTION
float distance(Point const &a, Point const &b)
{
  float distance_squared = 0.0;
  for (int d = 0; d < 3; ++d)
  {
    float tmp = b[d] - a[d];
    distance_squared += tmp * tmp;
  }
  return std::sqrt(distance_squared);
}

// distance point-box
KOKKOS_INLINE_FUNCTION
float distance(Point const &point, Box const &box)
{
  Point projected_point;
  for (int d = 0; d < 3; ++d)
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
KOKKOS_INLINE_FUNCTION
float distance(Point const &point, Sphere const &sphere)
{
  using KokkosExt::max;
  return max(distance(point, sphere.centroid()) - sphere.radius(), 0.f);
}

// distance box-box
KOKKOS_INLINE_FUNCTION
float distance(Box const &box_a, Box const &box_b)
{
  float distance_squared = 0.;
  for (int d = 0; d < 3; ++d)
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
KOKKOS_INLINE_FUNCTION
float distance(Sphere const &sphere, Box const &box)
{
  using KokkosExt::max;

  float distance_center_box = distance(sphere.centroid(), box);
  return max(distance_center_box - sphere.radius(), 0.f);
}

// expand an axis-aligned bounding box to include a point
KOKKOS_INLINE_FUNCTION
void expand(Box &box, Point const &point) { box += point; }

// expand an axis-aligned bounding box to include another box
// NOTE: Box type is templated here to be able to use expand(box, box) in a
// Kokkos::parallel_reduce() in which case the arguments must be declared
// volatile.
template <typename BOX,
          typename = typename std::enable_if<std::is_same<
              typename std::remove_volatile<BOX>::type, Box>::value>::type>
KOKKOS_INLINE_FUNCTION void expand(BOX &box, BOX const &other)
{
  box += other;
}

// expand an axis-aligned bounding box to include a sphere
KOKKOS_INLINE_FUNCTION
void expand(Box &box, Sphere const &sphere)
{
  using KokkosExt::max;
  using KokkosExt::min;
  for (int d = 0; d < 3; ++d)
  {
    box.minCorner()[d] =
        min(box.minCorner()[d], sphere.centroid()[d] - sphere.radius());
    box.maxCorner()[d] =
        max(box.maxCorner()[d], sphere.centroid()[d] + sphere.radius());
  }
}

// check if two axis-aligned bounding boxes intersect
KOKKOS_INLINE_FUNCTION
constexpr bool intersects(Box const &box, Box const &other)
{
  for (int d = 0; d < 3; ++d)
    if (box.minCorner()[d] > other.maxCorner()[d] ||
        box.maxCorner()[d] < other.minCorner()[d])
      return false;
  return true;
}

KOKKOS_INLINE_FUNCTION
constexpr bool intersects(Point const &point, Box const &other)
{
  for (int d = 0; d < 3; ++d)
    if (point[d] > other.maxCorner()[d] || point[d] < other.minCorner()[d])
      return false;
  return true;
}

// check if a sphere intersects with an  axis-aligned bounding box
KOKKOS_INLINE_FUNCTION
bool intersects(Sphere const &sphere, Box const &box)
{
  return distance(sphere.centroid(), box) <= sphere.radius();
}

KOKKOS_INLINE_FUNCTION
bool intersects(Sphere const &sphere, Point const &point)
{
  return distance(sphere.centroid(), point) <= sphere.radius();
}

KOKKOS_INLINE_FUNCTION
bool intersects(Point const &point, Sphere const &sphere)
{
  return intersects(sphere, point);
}

// calculate the centroid of a box
KOKKOS_INLINE_FUNCTION
void centroid(Box const &box, Point &c)
{
  for (int d = 0; d < 3; ++d)
    c[d] = (box.minCorner()[d] + box.maxCorner()[d]) / 2;
}

KOKKOS_INLINE_FUNCTION
void centroid(Point const &point, Point &c) { c = point; }

KOKKOS_INLINE_FUNCTION
void centroid(Sphere const &sphere, Point &c) { c = sphere.centroid(); }

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Point const &point) { return point; }

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Box const &box)
{
  Point c;
  for (int d = 0; d < 3; ++d)
    c[d] = (box.minCorner()[d] + box.maxCorner()[d]) / 2;
  return c;
}

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Sphere const &sphere) { return sphere.centroid(); }

// transformation that maps the unit cube into a new axis-aligned box
// NOTE safe to perform in-place
KOKKOS_INLINE_FUNCTION
void translateAndScale(Point const &in, Point &out, Box const &ref)
{
  for (int d = 0; d < 3; ++d)
  {
    auto const a = ref.minCorner()[d];
    auto const b = ref.maxCorner()[d];
    out[d] = (a != b ? (in[d] - a) / (b - a) : 0);
  }
}

} // namespace Details
} // namespace ArborX

#endif
