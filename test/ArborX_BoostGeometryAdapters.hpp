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
#include <ArborX_HyperBox.hpp>
#include <ArborX_HyperPoint.hpp>

#include <boost/geometry.hpp>

namespace boost
{
namespace geometry
{
namespace traits
{

// Adapt ArborX::ExperimentalHyperGeometry::Point to Boost.Geometry
template <int DIM, typename Coordinate>
struct tag<ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
{
  using type = point_tag;
};

template <int DIM, typename Coordinate>
struct coordinate_type<
    ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
{
  using type = Coordinate;
};

template <int DIM, typename Coordinate>
struct coordinate_system<
    ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
{
  using type = cs::cartesian;
};

template <int DIM, typename Coordinate>
struct dimension<ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
    : boost::mpl::int_<DIM>
{};

template <int DIM, typename Coordinate, size_t D>
struct access<ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>, D>
{
  static inline float
  get(ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate> const &p)
  {
    return p[D];
  }

  static inline void
  set(ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate> &p, float value)
  {
    p[D] = value;
  }
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
  using type = ArborX::ExperimentalHyperGeometry::Point<3>;
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

// Adapt ArborX::ExperimentalHyperGeometry::Box to Boost.Geometry
template <int DIM, typename Coordinate>
struct tag<ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  using type = box_tag;
};

template <int DIM, typename Coordinate>
struct point_type<ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  using type = ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>;
};

template <int DIM, typename Coordinate, size_t D>
struct indexed_access<ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>,
                      min_corner, D>
{
  static inline float
  get(ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate> const &b)
  {
    return b.minCorner()[D];
  }

  static inline void
  set(ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate> &b, float value)
  {
    b.minCorner()[D] = value;
  }
};

template <int DIM, typename Coordinate, size_t D>
struct indexed_access<ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>,
                      max_corner, D>
{
  static inline float
  get(ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate> const &b)
  {
    return b.maxCorner()[D];
  }

  static inline void
  set(ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate> &b, float value)
  {
    b.maxCorner()[D] = value;
  }
};

} // namespace traits
} // namespace geometry
} // namespace boost

#endif
