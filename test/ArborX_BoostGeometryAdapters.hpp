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

#ifndef ARBORX_BOOST_GEOMETRY_ADAPTERS_HPP
#define ARBORX_BOOST_GEOMETRY_ADAPTERS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Point.hpp>

#include <boost/geometry.hpp>

namespace boost
{
namespace geometry
{
namespace traits
{
// Adapt ArborX::Point to Boost.Geometry
template <>
struct tag<ArborX::Point>
{
  using type = point_tag;
};

template <>
struct coordinate_type<ArborX::Point>
{
  using type = float;
};

template <>
struct coordinate_system<ArborX::Point>
{
  using type = cs::cartesian;
};

template <>
struct dimension<ArborX::Point> : boost::mpl::int_<3>
{};

template <size_t D>
struct access<ArborX::Point, D>
{
  static inline float get(ArborX::Point const &p) { return p[D]; }

  static inline void set(ArborX::Point &p, float value) { p[D] = value; }
};

// Adapt ArborX::Box to Boost.Geometry
template <>
struct tag<ArborX::Box>
{
  using type = box_tag;
};

template <>
struct point_type<ArborX::Box>
{
  using type = ArborX::Point;
};

template <size_t D>
struct indexed_access<ArborX::Box, min_corner, D>
{
  static inline float get(ArborX::Box const &b) { return b.minCorner()[D]; }

  static inline void set(ArborX::Box &b, float value)
  {
    b.minCorner()[D] = value;
  }
};

template <size_t D>
struct indexed_access<ArborX::Box, max_corner, D>
{
  static inline float get(ArborX::Box const &b) { return b.maxCorner()[D]; }

  static inline void set(ArborX::Box &b, float value)
  {
    b.maxCorner()[D] = value;
  }
};

} // namespace traits
} // namespace geometry
} // namespace boost

#endif
