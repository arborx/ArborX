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
  typedef point_tag type;
};

template <>
struct coordinate_type<ArborX::Point>
{
  typedef float type;
};

template <>
struct coordinate_system<ArborX::Point>
{
  typedef cs::cartesian type;
};

template <>
struct dimension<ArborX::Point> : boost::mpl::int_<3>
{
};

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
  typedef box_tag type;
};

template <>
struct point_type<ArborX::Box>
{
  typedef ArborX::Point type;
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
