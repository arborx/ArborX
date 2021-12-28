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
#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Triangle.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cmath>

namespace ArborX
{
namespace Experimental
{
struct Quaternion
{
  float r;
  float i;
  float j;
  float k;
};

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

KOKKOS_INLINE_FUNCTION Quaternion hamiltonProduct(Quaternion const &a,
                                                  Quaternion const &b)
{
  Quaternion res;
  res.r = a.r * b.r - a.i * b.i - a.j * b.j - a.k * b.k;
  res.i = a.r * b.i + a.i * b.r + a.j * b.k - a.k * b.j;
  res.j = a.r * b.j - a.i * b.k + a.j * b.r + a.k * b.i;
  res.k = a.r * b.k + a.i * b.j - a.j * b.i + a.k * b.r;

  return res;
}

KOKKOS_INLINE_FUNCTION Point rotatePoint(Point const &point,
                                         Quaternion const &rotation_quat)
{
  Quaternion p = {0, point[0], point[1], point[2]};
  Quaternion rotation_quat_inv = {rotation_quat.r, -rotation_quat.i,
                                  -rotation_quat.j, -rotation_quat.k};
  auto rotated_p =
      hamiltonProduct(hamiltonProduct(rotation_quat, p), rotation_quat_inv);

  return {rotated_p.i, rotated_p.j, rotated_p.k};
}

KOKKOS_INLINE_FUNCTION Quaternion computeQuaternionRotation(float const &angle,
                                                            Vector const &axis)
{
  Quaternion rotation;
  rotation.r = std::cos(angle / 2.f);
  auto sin = std::sin(angle / 2.f);
  rotation.i = sin * axis[0];
  rotation.j = sin * axis[1];
  rotation.k = sin * axis[2];

  return rotation;
}

KOKKOS_INLINE_FUNCTION bool rayEdgeIntersect(Ray const &ray,
                                             Point const &edge_vertex_1,
                                             Point const &edge_vertex_2,
                                             float &t)
{
  // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
  float x1 = ray.origin()[0];
  float y1 = ray.origin()[1];
  float x2 = ray.origin()[0] + ray.direction()[0];
  float y2 = ray.origin()[1] + ray.direction()[1];
  float x3 = edge_vertex_1[0];
  float y3 = edge_vertex_1[1];
  float x4 = edge_vertex_2[0];
  float y4 = edge_vertex_2[1];
  float det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  auto const epsilon = 0.000001f;
  if (det > -epsilon && det < epsilon)
  {
    return false;
  }
  t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det;

  if (t >= 0.)
  {
    float u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / det;
    if (u >= 0. && u <= 1.)
      return true;
  }
  return false;
}

// Möller–Trumbore intersection algorithm
KOKKOS_INLINE_FUNCTION bool rayTriangleIntersect(Ray const &ray,
                                                 Triangle const &triangle,
                                                 float &t, float &u, float &v)
{
  auto const ab = makeVector(triangle.a, triangle.b);
  auto const ac = makeVector(triangle.a, triangle.c);

  auto const p = crossProduct(ray.direction(), ac);
  auto const det = dotProduct(ab, p);

  auto const epsilon = 0.000001f;
  // If the determinant is negative the triangle is back-facing.
  // If the determinant is close to 0, the ray and the triangle are in the same
  // plane.
  if (det > -epsilon && det < epsilon)
  {
    // Check if the ray hits an edge of the triangle
    // We need to rotate the triangle and the ray to be in the z = constant
    // plane First compute the normal to the triangle: cross product of ab and
    // ac
    auto normal = crossProduct(ab, ac);
    Ray::normalize(normal);
    // Check if the rotation is necessary
    auto normal_x = normal[0];
    auto normal_y = normal[1];
    auto normal_z = normal[2];
    Point rotated_triangle_a;
    Point rotated_triangle_b;
    Point rotated_triangle_c;
    Ray rotated_ray;
    if ((normal_x > -epsilon && normal_x < epsilon) &&
        (normal_y > -epsilon && normal_y < epsilon) &&
        ((normal_z > 1. - epsilon && normal_z < 1. + epsilon) ||
         (normal_z > -1. - epsilon && normal_z < -1. + epsilon)))
    {
      rotated_triangle_a = triangle.a;
      rotated_triangle_b = triangle.b;
      rotated_triangle_c = triangle.c;
      rotated_ray = ray;
    }
    else
    {
      float angle = std::acos((double)normal[2]);
      auto axis = crossProduct(normal, {0., 0., 1.});
      Ray::normalize(axis);
      auto rotation_quat = computeQuaternionRotation(angle, axis);
      rotated_triangle_a = rotatePoint(triangle.a, rotation_quat);
      rotated_triangle_b = rotatePoint(triangle.b, rotation_quat);
      rotated_triangle_c = rotatePoint(triangle.c, rotation_quat);
      rotated_ray.origin() = rotatePoint(ray.origin(), rotation_quat);
      auto rotated_direction = rotatePoint(
          {ray.direction()[0], ray.direction()[1], ray.direction()[2]},
          rotation_quat);
      rotated_ray.direction() = {rotated_direction[0], rotated_direction[1],
                                 rotated_direction[2]};
    }

    // Check that they are in the same plane
    if (rotated_triangle_a[2] - rotated_ray.origin()[2] > -epsilon &&
        rotated_triangle_a[2] - rotated_ray.origin()[2] < epsilon)
    {
      constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
      // A ray can intersect multiple edges so we need to check all ray-edge
      // intersections to find the first one.

      // Intersection with ab
      float t_ab = inf;
      bool ab_intersect = rayEdgeIntersect(rotated_ray, rotated_triangle_a,
                                           rotated_triangle_b, t_ab);
      if (ab_intersect)
        t = t_ab;

      // Intersection with ac
      float t_ac = inf;
      bool ac_intersect = rayEdgeIntersect(rotated_ray, rotated_triangle_a,
                                           rotated_triangle_c, t_ac);
      if (ac_intersect)
        t = KokkosExt::min(t, t_ac);

      // Intersection with bc
      float t_bc = inf;
      bool bc_intersect = rayEdgeIntersect(rotated_ray, rotated_triangle_b,
                                           rotated_triangle_c, t);
      if (bc_intersect)
        t = KokkosExt::min(t, t_bc);

      if (ac_intersect || ac_intersect || bc_intersect)
      {
        return true;
      }
    }

    return false;
  }

  auto const inv_det = 1 / det;

  auto const s = makeVector(triangle.a, ray.origin());
  u = inv_det * dotProduct(s, p);
  if (u < 0.f || u > 1.f)
  {
    return false;
  }

  auto const q = crossProduct(s, ab);
  v = inv_det * dotProduct(ray.direction(), q);
  if (v < 0.f || u + v > 1.f)
  {
    return false;
  }

  t = inv_det * dotProduct(ac, q);

  if (t < -epsilon)
    return false;

  return true;
}

KOKKOS_INLINE_FUNCTION bool intersects(Ray const &ray, Triangle const &triangle)
{
  float t;
  float u;
  float v;
  return rayTriangleIntersect(ray, triangle, t, u, v);
}

} // namespace Experimental
} // namespace ArborX
#endif
