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
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Triangle.hpp>
#include <detail/ArborX_GeometryAlgorithms.hpp> // equal
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Assert.hpp> // KOKKOS_ASSERT
#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>
#include <Kokkos_Swap.hpp>

#include <cmath>

namespace ArborX::Experimental
{

struct Ray
{
  using Point = ArborX::Point<3>;
  using Vector = ArborX::Details::Vector<3>;

  Point _origin = {};
  Vector _direction = {};

  KOKKOS_DEFAULTED_FUNCTION
  constexpr Ray() = default;

  KOKKOS_FUNCTION
  Ray(Point const &origin, Vector const &direction)
      : _origin(origin)
      , _direction(direction)
  {
    // Normalize direction using higher precision. Using `float` by default
    // creates a large error in the norm that affects ray tracing for
    // triangles.
    _direction = Details::normalize<double>(_direction);
  }

  KOKKOS_FUNCTION
  constexpr Point &origin() { return _origin; }

  KOKKOS_FUNCTION
  constexpr Point const &origin() const { return _origin; }

  KOKKOS_FUNCTION
  constexpr Vector &direction() { return _direction; }

  KOKKOS_FUNCTION
  constexpr Vector const &direction() const { return _direction; }
};

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Ray const &l, Ray const &r)
{
  using ArborX::Details::equals;
  return equals(l.origin(), r.origin()) && l.direction() == r.direction();
}

KOKKOS_INLINE_FUNCTION
auto returnCentroid(Ray const &ray) { return ray.origin(); }

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
bool intersection(Ray const &ray, Box<3> const &box, float &tmin, float &tmax)
{
  auto const &min = box.minCorner();
  auto const &max = box.maxCorner();
  auto const &orig = ray.origin();
  auto const &dir = ray.direction();

  constexpr auto inf =
      Details::KokkosExt::ArithmeticTraits::infinity<float>::value;
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
bool intersects(Ray const &ray, Box<3> const &box)
{
  float tmin;
  float tmax;
  // intersects only if box is in front of the ray
  return intersection(ray, box, tmin, tmax) && (tmax >= 0.f);
}

// The function returns the index of the largest
// component of the direction vector.
KOKKOS_INLINE_FUNCTION int findLargestComp(typename Ray::Vector const &dir)
{
  int kz = 0;

  auto max = std::abs(dir[0]);

  for (int i = 1; i < 3; i++)
  {
    auto f = std::fabs(dir[i]);

    if (f > max)
    {
      max = f;
      kz = i;
    }
  }

  return kz;
}

// Both the ray and the triangle were transformed beforehand
// so that the ray is a unit vector along the z-axis (0,0,1),
// and the triangle is transformed with the same matrix
// (which is M in the paper). This function is called only
// when the ray is co-planar to the triangle (with the
// determinant being zero). The rotation by this function is
// to prepare for the ray-edge intersection calculations in 2D.
// The rotation is around the z-axis. For any point after
// the rotation, its new x* equals its original length
// with the correct sign, and the new y* = z. The current
// implementation avoids explicitly defining rotation angles
// and directions. The following ray-edge intersection will
// be in the x*-y* plane.
KOKKOS_INLINE_FUNCTION auto rotate2D(Ray::Point const &point)
{
  Ray::Point point_star;
  float r = std::sqrt(point[0] * point[0] + point[1] * point[1]);
  if (point[0] != 0)
  {
    point_star[0] = (point[0] > 0 ? 1 : -1) * r;
  }
  else
  {
    point_star[0] = (point[1] > 0 ? 1 : -1) * r;
  }
  point_star[1] = point[2];
  point_star[2] = 0.f;
  return point_star;
}

// The function is for ray-edge intersection
// with the rotated ray along the z-axis and
// the transformed and rotated triangle edges
// The algorithm is described in
// https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
KOKKOS_INLINE_FUNCTION bool rayEdgeIntersect(Ray::Point const &edge_vertex_1,
                                             Ray::Point const &edge_vertex_2,
                                             float &t)
{
  float x3 = edge_vertex_1[0];
  float y3 = edge_vertex_1[1];
  float x4 = edge_vertex_2[0];
  float y4 = edge_vertex_2[1];

  float y2 = std::fabs(y3) > std::fabs(y4) ? y3 : y4;

  float det = y2 * (x3 - x4);

  //  the ray is parallel to the edge if det == 0.0
  //  When the ray overlaps the edge (x3==x4==0.0), it also returns false,
  //  and the intersection will be captured by the other two edges.
  if (det == 0)
  {
    return false;
  }
  t = (x3 * y4 - x4 * y3) / det * y2;

  float u = x3 * y2 / det;

  auto const epsilon = 0.00001f;
  return (u >= 0 - epsilon && u <= 1 + epsilon);
}

// The algorithm is described in
// Watertight Ray/Triangle Intersection
// [1] Woop, S. et al. (2013),
// Journal of Computer Graphics Techniques Vol. 2(1)
// The major difference is that here we return the intersection points
// when the ray and the triangle is coplanar.
// In the paper, they just need the boolean return.
KOKKOS_INLINE_FUNCTION
bool intersection(Ray const &ray, Triangle<3> const &triangle, float &tmin,
                  float &tmax)
{
  namespace KokkosExt = Details::KokkosExt;

  using Vector = typename Ray::Vector;

  auto dir = ray.direction();
  // normalize the direction vector by its largest component.
  auto kz = findLargestComp(dir);
  int kx = (kz + 1) % 3;
  int ky = (kz + 2) % 3;

  if (dir[kz] < 0)
    Kokkos::kokkos_swap(kx, ky);

  Vector s;

  s[2] = 1.0f / dir[kz];
  s[0] = dir[kx] * s[2];
  s[1] = dir[ky] * s[2];

  // calculate vertices relative to ray origin
  auto const &o = ray.origin();
  auto const oA = triangle.a - o;
  auto const oB = triangle.b - o;
  auto const oC = triangle.c - o;

  // oA, oB, oB need to be normalized, otherwise they
  // will scale with the problem size.
  auto const mag_oA = oA.norm();
  auto const mag_oB = oB.norm();
  auto const mag_oC = oC.norm();

  auto mag_bar = 3.0 / (mag_oA + mag_oB + mag_oC);

  Ray::Point A;
  Ray::Point B;
  Ray::Point C;

  // perform shear and scale of vertices
  // normalized by mag_bar
  A[0] = (oA[kx] - s[0] * oA[kz]) * mag_bar;
  A[1] = (oA[ky] - s[1] * oA[kz]) * mag_bar;
  B[0] = (oB[kx] - s[0] * oB[kz]) * mag_bar;
  B[1] = (oB[ky] - s[1] * oB[kz]) * mag_bar;
  C[0] = (oC[kx] - s[0] * oC[kz]) * mag_bar;
  C[1] = (oC[ky] - s[1] * oC[kz]) * mag_bar;

  // calculate scaled barycentric coordinates
  float u = C[0] * B[1] - C[1] * B[0];
  float v = A[0] * C[1] - A[1] * C[0];
  float w = B[0] * A[1] - B[1] * A[0];

  // fallback to double precision
  if (u == 0 || v == 0 || w == 0)
  {
    u = (double)C[0] * B[1] - (double)C[1] * B[0];
    v = (double)A[0] * C[1] - (double)A[1] * C[0];
    w = (double)B[0] * A[1] - (double)B[1] * A[0];
  }

  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = inf;
  tmax = -inf;

  // 'Back-face culling' is not supported.
  // Back-facing culling is to check whether
  // a surface is 'visible' to a ray, which requires
  // consistent definition of the facing of triangles.
  // Once the facing of triangle is defined,
  // only one of the conditions is needed,
  // either (u < 0 || v < 0 || w < 0) or
  // (u > 0 || v > 0 || w > 0), for Back-facing culling.
  float const epsilon = 0.0000001f;
  if ((u < -epsilon || v < -epsilon || w < -epsilon) &&
      (u > epsilon || v > epsilon || w > epsilon))
    return false;

  // calculate determinant
  float det = u + v + w;

  A[2] = s[2] * oA[kz];
  B[2] = s[2] * oB[kz];
  C[2] = s[2] * oC[kz];

  if (det < -epsilon || det > epsilon)
  {
    float t = (u * A[2] + v * B[2] + w * C[2]) / det;
    tmax = t;
    tmin = t;
    return true;
  }

  // The ray is co-planar to the triangle.
  // Check the intersection with each edge
  // the rotate2D function is to make sure the ray-edge
  // intersection check is at the plane where ray and edges
  // are at.
  auto A_star = rotate2D(A);
  auto B_star = rotate2D(B);
  auto C_star = rotate2D(C);

  float t_ab = inf;
  bool ab_intersect = rayEdgeIntersect(A_star, B_star, t_ab);
  if (ab_intersect)
  {
    tmin = t_ab;
    tmax = t_ab;
  }
  float t_bc = inf;
  bool bc_intersect = rayEdgeIntersect(B_star, C_star, t_bc);
  if (bc_intersect)
  {
    tmin = Kokkos::min(tmin, t_bc);
    tmax = Kokkos::max(tmax, t_bc);
  }
  float t_ca = inf;
  bool ca_intersect = rayEdgeIntersect(C_star, A_star, t_ca);
  if (ca_intersect)
  {
    tmin = Kokkos::min(tmin, t_ca);
    tmax = Kokkos::max(tmax, t_ca);
  }

  if (ab_intersect || bc_intersect || ca_intersect)
  {
    // When (1) the origin of the ray is within the triangle
    // and (2) they ray is coplanar with the triangle, the
    // intersection length is zero.
    if (tmin * tmax <= 0)
    {
      tmin = 0;
      tmax = 0;
    }
    else
    {
      // need to separate tmin tmax >0 and <0 cases
      // e.g., tmin = -2 and tmax = -1, but
      // we want tmin = -1 and tmax = -2, when the
      // ray travels backward
      if (tmin < 0)
        Kokkos::kokkos_swap(tmin, tmax);
    }
    return true;
  }

  return false;
} // namespace Experimental

KOKKOS_INLINE_FUNCTION bool intersects(Ray const &ray,
                                       Triangle<3> const &triangle)
{
  float tmin;
  float tmax;
  // intersects only if triangle is in front of the ray
  return intersection(ray, triangle, tmin, tmax) && (tmax >= 0.f);
}

// Returns the first positive value for t such that ray.origin + t * direction
// intersects the given box. If no such value exists, returns inf.
// Note that this definiton is different from the standard
// "smallest distance between a point on the ray and a point in the box"
// so we can use nearest queries for ray tracing.
KOKKOS_INLINE_FUNCTION
float distance(Ray const &ray, Box<3> const &box)
{
  float tmin;
  float tmax;
  bool intersects = intersection(ray, box, tmin, tmax) && (tmax >= 0.f);
  return intersects
             ? (tmin > 0.f ? tmin : 0.f)
             : Details::KokkosExt::ArithmeticTraits::infinity<float>::value;
}

// Solves a*x^2 + b*x + c = 0.
// If a solution exists, return true and stores roots at x1, x2.
// If a solution does not exist, returns false.
KOKKOS_INLINE_FUNCTION bool solveQuadratic(float const a, float const b,
                                           float const c, float &x1, float &x2)
{
  KOKKOS_ASSERT(a != 0);

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
KOKKOS_INLINE_FUNCTION bool
intersection(Ray const &ray, Sphere<3> const &sphere, float &tmin, float &tmax)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;

  auto const &r = sphere.radius();

  // Vector oc = (origin_of_ray - center_of_sphere)
  auto const oc = ray.origin() - sphere.centroid();

  float const a2 = 1.f; // directions are normalized
  float const a1 = 2.f * oc.dot(ray.direction());
  float const a0 = oc.dot(oc) - r * r;

  if (solveQuadratic(a2, a1, a0, tmin, tmax))
  {
    // ensures that tmin <= tmax
    if (tmin > tmax)
      Kokkos::kokkos_swap(tmin, tmax);

    return true;
  }
  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = inf;
  tmax = -inf;
  return false;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION void
overlapDistance(Ray const &ray, Geometry const &geometry, float &length,
                float &distance_to_origin)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;

  float tmin;
  float tmax;
  if (intersection(ray, geometry, tmin, tmax) && (tmin <= tmax && tmax >= 0))
  {
    // Overlap [tmin, tmax] with [0, +inf)
    tmin = Kokkos::max(0.f, tmin);
    // As direction is normalized,
    //   |(o + tmax*d) - (o + tmin*d)| = tmax - tmin
    length = tmax - tmin;
    distance_to_origin = tmin;
  }
  else
  {
    length = 0;
    distance_to_origin = KokkosExt::ArithmeticTraits::infinity<float>::value;
  }
}

KOKKOS_INLINE_FUNCTION float overlapDistance(Ray const &ray,
                                             Sphere<3> const &sphere)
{
  float distance_to_origin;
  float length;
  overlapDistance(ray, sphere, length, distance_to_origin);
  return length;
}

} // namespace ArborX::Experimental

template <>
struct ArborX::GeometryTraits::dimension<ArborX::Experimental::Ray>
{
  static constexpr int value = 3;
};
template <>
struct ArborX::GeometryTraits::tag<ArborX::Experimental::Ray>
{
  using type = RayTag;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<ArborX::Experimental::Ray>
{
  using type = float;
};

#endif
