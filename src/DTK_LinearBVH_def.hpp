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

#ifndef DTK_LINEAR_BVH_DEF_HPP
#define DTK_LINEAR_BVH_DEF_HPP

#include <DTK_Box.hpp>
#include <DTK_DetailsConcepts.hpp>
#include <DTK_Point.hpp>

namespace DataTransferKit
{

// FIXME nothing here...

// FIXME not sure where to put these
static_assert( Details::is_expandable<Box, Box>::value, "" );
static_assert( Details::is_expandable<Box, Box const>::value, "" );
static_assert( Details::is_expandable<Box, Point>::value, "" );
static_assert( Details::is_expandable<Box, Point const>::value, "" );
static_assert( Details::has_centroid<Box, Point>::value, "" );
static_assert( Details::has_centroid<Box const, Point>::value, "" );
static_assert( Details::has_centroid<Point, Point>::value, "" );
static_assert( Details::has_centroid<Point const, Point>::value, "" );

} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_LINEAR_BVH_INSTANT( NODE )                                         \
    template class BoundingVolumeHierarchy<typename NODE::device_type>;

#endif
