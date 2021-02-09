/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_RAY_HPP
#define ARBORX_RAY_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // equal
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cmath>

namespace ArborX
{
namespace Experimental
{
struct Ray
{
  using Vector = Point; // will regret this later

  using Scalar = std::decay_t<decltype(std::declval<Vector>()[0])>;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr Ray() = default;

  KOKKOS_FUNCTION
  Ray(Point const &origin, Vector const &direction)
      : _origin(origin)
      , _direction(direction)
  {
    normalize(_direction);
  }

  KOKKOS_FUNCTION
  static Scalar norm(Vector const &v)
  {
    Scalar sq{};
    for (int d = 0; d < 3; ++d)
      sq += v[d] * v[d];
    return std::sqrt(sq);
  }

  KOKKOS_FUNCTION static void normalize(Vector &v)
  {
    auto const magv = norm(v);
    assert(magv > 0);
    for (int d = 0; d < 3; ++d)
      v[d] /= magv;
  }

  KOKKOS_FUNCTION
  constexpr Point &origin() { return _origin; }

  KOKKOS_FUNCTION
  constexpr Point const &origin() const { return _origin; }

  KOKKOS_FUNCTION
  constexpr Vector &direction() { return _direction; }

  KOKKOS_FUNCTION
  constexpr Vector const &direction() const { return _direction; }

  Point _origin = {};
  Vector _direction = {{0.f, 0.f, 0.f}};
};

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Ray const &l, Ray const &r)
{
  using ArborX::Details::equals;
  return equals(l.origin(), r.origin()) && equals(l.direction(), r.direction());
}

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Ray const &ray) { return ray.origin(); }

// The ray-box intersection algorithm is based on [1]. Their 'efficient slag'
// algorithm checks the intersections both in front and behind the ray. The
// function here checks the intersections in front of the ray.
//
// There are few issues here. First, when a ray direction is aligned with one
// of the axis, a division by zero will occur. This is fine, as usually it
// results in +inf or -inf, which are treated correctly. However, it also leads
// to the second situation, when it is 0/0 which occurs when the ray's origin
// in that dimension is on the same plane as one of the corners (i.e., if
// inv_ray_dir[d] == 0 && (minCorner[d] == origin[d] || maxCorner[d] ==
// origin[d])). This leads to NaN, which are not treated correctly (unless, as
// in [1], the underlying min/max functions are able to ignore them). The issue
// is discussed in more details in [2] and the webiste (key word: A minimal
// ray-tracer: rendering simple shapes).
//
// In the algorithm below, we explicitly ignoring NaN values, leading to
// correct algorithm. An interesting side note is that per IEEE standard, all
// comparisons with NaN are false.
//
// [1] Majercik, A., Crassin, C., Shirley, P., & McGuire, M. (2018). A ray-box
// intersection algorithm and efficient dynamic voxel rendering. Journal of
// Computer Graphics Techniques Vol, 7(3).
//
// [2] Williams, A., Barrus, S., Morley, R. K., & Shirley, P. (2005). An
// efficient and robust ray-box intersection algorithm. In ACM SIGGRAPH 2005
// Courses (pp. 9-es).
KOKKOS_INLINE_FUNCTION
bool intersects(Ray const &ray, Box const &box)
{
  auto const &minCorner = box.minCorner();
  auto const &maxCorner = box.maxCorner();
  auto const &origin = ray.origin();
  auto const &direction = ray.direction();

  auto const inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  float max_min = -inf;
  float min_max = inf;

  for (int d = 0; d < 3; ++d)
  {
    float tmin;
    float tmax;
    if (direction[d] >= 0)
    {
      tmin = (minCorner[d] - origin[d]) / direction[d];
      tmax = (maxCorner[d] - origin[d]) / direction[d];
    }
    else
    {
      tmin = (maxCorner[d] - origin[d]) / direction[d];
      tmax = (minCorner[d] - origin[d]) / direction[d];
    }

    if (!std::isnan(tmin) && max_min < tmin)
      max_min = tmin;
    if (!std::isnan(tmax) && min_max > tmax)
      min_max = tmax;
  }

  return max_min <= min_max && (min_max >= 0);
}

KOKKOS_INLINE_FUNCTION float dotProduct(Ray::Vector const &v1,
                                        Ray::Vector const &v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// Solves a*x^2 + b*x + c = 0.
// If a solution exists, return true and stores roots at x1, x2.
// If a solution does not exist, returns false.
KOKKOS_INLINE_FUNCTION bool solveQuadratic(float const a, float const b,
                                           float const c, float &x1, float &x2)
{
  assert(a != 0);

  auto const discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
    return false;
  if (discriminant == 0)
  {
    x1 = x2 = -b / (2 * a);
    return true;
  }

  // Instead of doing a simple
  //    (-b +- std::sqrt(discriminant)) / (2*a)
  // we use a more stable algorithm with less loss of precision (see, for
  // example, https://www.scratchapixel.com/lessons/3d-basic-rendering/
  // minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection).
  auto const q = (b > 0) ? (-b - std::sqrt(discriminant)) / 2.0
                         : (-b + std::sqrt(discriminant)) / 2.0;
  x1 = q / a;
  x2 = c / q;

  return true;
}

// Ray-Sphere intersection algorithm.
//
// The sphere can be expressed as the solution to
//     |p - c|^2 - r^2 = 0,           (1)
// where c is the center of the sphere, and r is the radius. On the other
// hand, any point on a bidirectional ray satisfies
//     p = o + t*d,                   (2)
// where o is the origin, and d is the direction vector.
// Substituting (2) into (1),
//     |(o + t*d) - c|^2 - r^2 = 0,   (3)
// results in a quadratic equation for unknown t
//     a2 * t^2 + a1 * t + a0 = 0
// with
//     a2 = |d|^2, a1 = 2*(d, o - c), and a0 = |o - c|^2 - r^2.
// Then, we only need to intersect the solution interval [tmin, tmax] with
// [0, +inf) for the unidirectional ray.
KOKKOS_INLINE_FUNCTION float overlapDistance(Ray const &ray,
                                             Sphere const &sphere)
{
  auto const &r = sphere.radius();

  // Vector oc = (origin_of_ray - center_of_sphere)
  Ray::Vector const oc{ray.origin()[0] - sphere.centroid()[0],
                       ray.origin()[1] - sphere.centroid()[1],
                       ray.origin()[2] - sphere.centroid()[2]};

  float const a2 = 1.f; // directions are normalized
  float const a1 = 2.f * dotProduct(ray.direction(), oc);
  float const a0 = dotProduct(oc, oc) - r * r;

  float t1;
  float t2;
  if (!solveQuadratic(a2, a1, a0, t1, t2))
  {
    // No intersection of a bidirectional ray with the sphere
    return 0.f;
  }
  float tmin = KokkosExt::min(t1, t2);
  float tmax = KokkosExt::max(t1, t2);

  if (tmax < 0)
  {
    // Half-ray does not intersect with the sphere
    return 0.f;
  }

  // Overlap [tmin, tmax] with [0, +inf)
  tmin = KokkosExt::max(0.f, tmin);

  // As direction is normalized,
  //   |(o + tmax*d) - (o + tmin*d)| = tmax - tmin
  return (tmax - tmin);
}

} // namespace Experimental
} // namespace ArborX
#endif
