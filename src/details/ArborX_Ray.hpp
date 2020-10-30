/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
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
#include <ArborX_DetailsKokkosExt.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cmath>

namespace ArborX
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

namespace Details
{
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

} // namespace Details
} // namespace ArborX
#endif
