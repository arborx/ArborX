/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_POINT_HPP
#define DTK_POINT_HPP

#include <Kokkos_Macros.hpp>

namespace DataTransferKit
{
class Point
{
  public:
    KOKKOS_INLINE_FUNCTION
    const double &operator()( unsigned int const i ) const
    {
        return _coords[i];
    }

    KOKKOS_INLINE_FUNCTION
    const double &operator[]( unsigned int i ) const { return _coords[i]; }

    KOKKOS_INLINE_FUNCTION
    double &operator()( unsigned int const i ) { return _coords[i]; }

    KOKKOS_INLINE_FUNCTION
    double &operator[]( unsigned int i ) { return _coords[i]; }

    // This should be private but if we make public we can use the list
    // initializer constructor.
    double _coords[3];
};
}

#endif
