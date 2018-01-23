/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_DETAILS_POINT_HPP
#define DTK_DETAILS_POINT_HPP

#include <Kokkos_Macros.hpp>

namespace DataTransferKit
{
class Point
{
  public:
    KOKKOS_INLINE_FUNCTION
    double &operator[]( unsigned int i ) { return _coords[i]; }

    KOKKOS_INLINE_FUNCTION
    double const &operator[]( unsigned int i ) const { return _coords[i]; }

    KOKKOS_INLINE_FUNCTION
    double volatile &operator[]( unsigned int i ) volatile
    {
        return _coords[i];
    }

    KOKKOS_INLINE_FUNCTION
    double const volatile &operator[]( unsigned int i ) const volatile
    {
        return _coords[i];
    }

    // This should be private but if we make public we can use the list
    // initializer constructor.
    double _coords[3];
};
}

#endif
