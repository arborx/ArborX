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
#ifndef ARBORX_RAY_HPP
#define ARBORX_RAY_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // equal
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cmath>

namespace ArborX
{
namespace Experimental
{

struct Vector : private Point
{
  using Point::Point;
  using Point::operator[];
  friend KOKKOS_FUNCTION constexpr bool operator==(Vector const &v,
                                                   Vector const &w)
  {
    return v[0] == w[0] && v[1] == w[1] && v[2] == w[2];
  }
};

KOKKOS_INLINE_FUNCTION constexpr Vector makeVector(Point const &begin,
                                                   Point const &end)
{
  Vector v;
  for (int d = 0; d < 3; ++d)
  {
    v[d] = end[d] - begin[d];
  }
  return v;
}

KOKKOS_INLINE_FUNCTION constexpr auto dotProduct(Vector const &v,
                                                 Vector const &w)
{
  return v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
}

KOKKOS_INLINE_FUNCTION constexpr Vector crossProduct(Vector const &v,
                                                     Vector const &w)
{
  return {v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2],
          v[0] * w[1] - v[1] * w[0]};
}

KOKKOS_INLINE_FUNCTION constexpr bool equals(Vector const &v, Vector const &w)
{
  return v == w;
}

struct Ray
{
  Point _origin = {};
  Vector _direction = {};

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
  constexpr Point &origin() { return _origin; }

  KOKKOS_FUNCTION
  constexpr Point const &origin() const { return _origin; }

  KOKKOS_FUNCTION
  constexpr Vector &direction() { return _direction; }

  KOKKOS_FUNCTION
  constexpr Vector const &direction() const { return _direction; }

private:
  // We would like to use Scalar defined as:
  // using Scalar = std::decay_t<decltype(std::declval<Vector>()[0])>;
  // However, this means using float to compute the norm. This creates a large
  // error in the norm that affects ray tracing for triangles. Casting the
  // norm from double to float once it has been computed is not enough to
  // improve the value of the normalized vector. Thus, the norm has to return a
  // double.
  using Scalar = double;

  KOKKOS_FUNCTION
  static Scalar norm(Vector const &v)
  {
    Scalar sq{};
    for (int d = 0; d < 3; ++d)
      sq += static_cast<Scalar>(v[d]) * static_cast<Scalar>(v[d]);
    return std::sqrt(sq);
  }

  KOKKOS_FUNCTION static void normalize(Vector &v)
  {
    auto const magv = norm(v);
    assert(magv > 0);
    for (int d = 0; d < 3; ++d)
      v[d] /= magv;
  }
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
// algorithm checks the intersections both in front and behind the ray.
//
// There are few issues here. First, when a ray direction is aligned with one
// of the axis, a division by zero will occur. This is fine, as usually it
// results in +inf or -inf, which are treated correctly. However, it also leads
// to the second situation, when it is 0/0 which occurs when the ray's origin
// in that dimension is on the same plane as one of the corners of the box
// (i.e., if inv_ray_dir[d] == 0 && (min_corner[d] == origin[d] || max_corner[d]
// == origin[d])). This leads to NaN, which are not treated correctly (unless,
// as in [1], the underlying min/max functions are able to ignore them). The
// issue is discussed in more details in [2] and the website (key word: A
// minimal ray-tracer: rendering simple shapes).
//
// [1] Majercik, A., Crassin, C., Shirley, P., & McGuire, M. (2018). A ray-box
// intersection algorithm and efficient dynamic voxel rendering. Journal of
// Computer Graphics Techniques Vol, 7(3).
//
// [2] Williams, A., Barrus, S., Morley, R. K., & Shirley, P. (2005). An
// efficient and robust ray-box intersection algorithm. In ACM SIGGRAPH 2005
// Courses (pp. 9-es).
KOKKOS_INLINE_FUNCTION
bool intersection(Ray const &ray, Box const &box, float &tmin, float &tmax)
{
  auto const &min = box.minCorner();
  auto const &max = box.maxCorner();
  auto const &orig = ray.origin();
  auto const &dir = ray.direction();

  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = -inf;
  tmax = inf;

  for (int d = 0; d < 3; ++d)
  {
    float tdmin;
    float tdmax;
    if (dir[d] >= 0)
    {
      tdmin = (min[d] - orig[d]) / dir[d];
      tdmax = (max[d] - orig[d]) / dir[d];
    }
    else
    {
      tdmin = (max[d] - orig[d]) / dir[d];
      tdmax = (min[d] - orig[d]) / dir[d];
    }
    if (tmin < tdmin)
      tmin = tdmin;
    if (tmax > tdmax)
      tmax = tdmax;
  }
  return (tmin <= tmax);
}

KOKKOS_INLINE_FUNCTION
bool intersects(Ray const &ray, Box const &box)
{
  float tmin;
  float tmax;
  // intersects only if box is in front of the ray
  return intersection(ray, box, tmin, tmax) && (tmax >= 0.f);
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
  // clang-format off
  // example, https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection).
  // clang-format on
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
KOKKOS_INLINE_FUNCTION bool intersection(Ray const &ray, Sphere const &sphere,
                                         float &tmin, float &tmax)
{
  auto const &r = sphere.radius();

  // Vector oc = (origin_of_ray - center_of_sphere)
  Vector const oc = makeVector(sphere.centroid(), ray.origin());

  float const a2 = 1.f; // directions are normalized
  float const a1 = 2.f * dotProduct(ray.direction(), oc);
  float const a0 = dotProduct(oc, oc) - r * r;

  if (solveQuadratic(a2, a1, a0, tmin, tmax))
  {
    // ensures that tmin <= tmax
    if (tmin > tmax)
      KokkosExt::swap(tmin, tmax);

    return true;
  }
  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = inf;
  tmax = -inf;
  return false;
}

KOKKOS_INLINE_FUNCTION float overlapDistance(Ray const &ray,
                                             Sphere const &sphere)
{
  float tmin;
  float tmax;
  if (!intersection(ray, sphere, tmin, tmax) || (tmax < 0))
  {
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
