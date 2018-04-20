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
#ifndef DTK_BOX_HPP
#define DTK_BOX_HPP

#include <DTK_Point.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Macros.hpp>

namespace DataTransferKit
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
struct Box
{
    KOKKOS_INLINE_FUNCTION
    Box() = default;

    KOKKOS_INLINE_FUNCTION
    Box( Point const &min_corner, Point const &max_corner )
        : _min_corner( min_corner )
        , _max_corner( max_corner )
    {
    }

    KOKKOS_INLINE_FUNCTION
    Point &minCorner() { return _min_corner; }

    KOKKOS_INLINE_FUNCTION
    Point const &minCorner() const { return _min_corner; }

    KOKKOS_INLINE_FUNCTION
    Point volatile &minCorner() volatile { return _min_corner; }

    KOKKOS_INLINE_FUNCTION
    Point volatile const &minCorner() volatile const { return _min_corner; }

    KOKKOS_INLINE_FUNCTION
    Point &maxCorner() { return _max_corner; }

    KOKKOS_INLINE_FUNCTION
    Point const &maxCorner() const { return _max_corner; }

    KOKKOS_INLINE_FUNCTION
    Point volatile &maxCorner() volatile { return _max_corner; }

    KOKKOS_INLINE_FUNCTION
    Point volatile const &maxCorner() volatile const { return _max_corner; }

    Point _min_corner = {{Kokkos::ArithTraits<double>::max(),
                          Kokkos::ArithTraits<double>::max(),
                          Kokkos::ArithTraits<double>::max()}};
    Point _max_corner = {{-Kokkos::ArithTraits<double>::max(),
                          -Kokkos::ArithTraits<double>::max(),
                          -Kokkos::ArithTraits<double>::max()}};
};
} // namespace DataTransferKit

#endif
