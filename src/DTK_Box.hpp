/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_Box_HPP
#define DTK_Box_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Core.hpp>

namespace DataTransferKit
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
struct Box
{
    using ArrayType = double[6]; // Kokkos::Array<double, 6>;
    using SizeType = size_t;     // ArrayType::size_type;

    KOKKOS_INLINE_FUNCTION
    Box()
    {
        _minmax[0] = Kokkos::ArithTraits<double>::max();
        _minmax[1] = -Kokkos::ArithTraits<double>::max();
        _minmax[2] = Kokkos::ArithTraits<double>::max();
        _minmax[3] = -Kokkos::ArithTraits<double>::max();
        _minmax[4] = Kokkos::ArithTraits<double>::max();
        _minmax[5] = -Kokkos::ArithTraits<double>::max();
    }

    KOKKOS_INLINE_FUNCTION
    Box( ArrayType const &minmax )
    {
        for ( unsigned int i = 0; i < 6; ++i )
            _minmax[i] = minmax[i];
    }

    KOKKOS_INLINE_FUNCTION
    Box &operator=( ArrayType const &minmax )
    {
        for ( unsigned int i = 0; i < 6; ++i )
            _minmax[i] = minmax[i];

        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    double &operator[]( SizeType i ) { return _minmax[i]; }

    KOKKOS_INLINE_FUNCTION
    double const &operator[]( SizeType i ) const { return _minmax[i]; }

    KOKKOS_INLINE_FUNCTION
    volatile double &operator[]( SizeType i ) volatile { return _minmax[i]; }

    KOKKOS_INLINE_FUNCTION
    volatile double const &operator[]( SizeType i ) volatile const
    {
        return _minmax[i];
    }

    ArrayType _minmax;

    friend std::ostream &operator<<( std::ostream &os, Box const &aabb )
    {
        os << "{";
        for ( int d = 0; d < 3; ++d )
            os << " [" << aabb[2 * d + 0] << ", " << aabb[2 * d + 1] << "],";
        os << "}";
        return os;
    }
};
}

#endif
