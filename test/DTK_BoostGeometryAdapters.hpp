/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_BOOST_GEOMETRY_ADAPTERS_HPP
#define DTK_BOOST_GEOMETRY_ADAPTERS_HPP

#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsPoint.hpp>

#include <boost/geometry.hpp>

namespace boost
{
namespace geometry
{
namespace traits
{
// Adapt DataTransferKit::Point to Boost.Geometry
template <>
struct tag<DataTransferKit::Point>
{
    typedef point_tag type;
};

template <>
struct coordinate_type<DataTransferKit::Point>
{
    typedef double type;
};

template <>
struct coordinate_system<DataTransferKit::Point>
{
    typedef cs::cartesian type;
};

template <>
struct dimension<DataTransferKit::Point> : boost::mpl::int_<3>
{
};

template <size_t D>
struct access<DataTransferKit::Point, D>
{
    static inline double get( DataTransferKit::Point const &p ) { return p[D]; }

    static inline void set( DataTransferKit::Point &p, double value )
    {
        p[D] = value;
    }
};

// Adapt DataTransferKit::Box to Boost.Geometry
template <>
struct tag<DataTransferKit::Box>
{
    typedef box_tag type;
};

template <>
struct point_type<DataTransferKit::Box>
{
    typedef DataTransferKit::Point type;
};

template <size_t D>
struct indexed_access<DataTransferKit::Box, min_corner, D>
{
    static inline double get( DataTransferKit::Box const &b )
    {
        return b.minCorner()[D];
    }

    static inline void set( DataTransferKit::Box &b, double value )
    {
        b.minCorner()[D] = value;
    }
};

template <size_t D>
struct indexed_access<DataTransferKit::Box, max_corner, D>
{
    static inline double get( DataTransferKit::Box const &b )
    {
        return b.maxCorner()[D];
    }

    static inline void set( DataTransferKit::Box &b, double value )
    {
        b.maxCorner()[D] = value;
    }
};

} // end namespace traits
} // end namespace geometry
} // end namespace boost

#endif
