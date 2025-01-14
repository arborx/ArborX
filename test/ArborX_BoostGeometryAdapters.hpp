/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
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

namespace boost::geometry::traits
{

// Adapt ArborX::Point to Boost.Geometry
template <int DIM, typename Coordinate>
struct tag<ArborX::Point<DIM, Coordinate>>
{
  using type = point_tag;
};

template <int DIM, typename Coordinate>
struct coordinate_type<ArborX::Point<DIM, Coordinate>>
{
  using type = Coordinate;
};

template <int DIM, typename Coordinate>
struct coordinate_system<ArborX::Point<DIM, Coordinate>>
{
  using type = cs::cartesian;
};

template <int DIM, typename Coordinate>
struct dimension<ArborX::Point<DIM, Coordinate>> : boost::mpl::int_<DIM>
{};

template <int DIM, typename Coordinate, size_t D>
struct access<ArborX::Point<DIM, Coordinate>, D>
{
  static inline float get(ArborX::Point<DIM, Coordinate> const &p)
  {
    return p[D];
  }

  static inline void set(ArborX::Point<DIM, Coordinate> &p, float value)
  {
    p[D] = value;
  }
};

// Adapt ArborX::Box to Boost.Geometry
template <int DIM, typename Coordinate>
struct tag<ArborX::Box<DIM, Coordinate>>
{
  using type = box_tag;
};

template <int DIM, typename Coordinate>
struct point_type<ArborX::Box<DIM, Coordinate>>
{
  using type = ArborX::Point<DIM, Coordinate>;
};

template <int DIM, typename Coordinate, size_t D>
struct indexed_access<ArborX::Box<DIM, Coordinate>, min_corner, D>
{
  static inline float get(ArborX::Box<DIM, Coordinate> const &b)
  {
    return b.minCorner()[D];
  }

  static inline void set(ArborX::Box<DIM, Coordinate> &b, float value)
  {
    b.minCorner()[D] = value;
  }
};

template <int DIM, typename Coordinate, size_t D>
struct indexed_access<ArborX::Box<DIM, Coordinate>, max_corner, D>
{
  static inline float get(ArborX::Box<DIM, Coordinate> const &b)
  {
    return b.maxCorner()[D];
  }

  static inline void set(ArborX::Box<DIM, Coordinate> &b, float value)
  {
    b.maxCorner()[D] = value;
  }
};

} // namespace boost::geometry::traits

#endif
